"""
support.py - The fixtures the database-backed search steps need.

calculate_fitness, elitism, selection and mutation are all written as
`step(conn, run_id, ...)`, so testing them means handing them a sweep. This
builds one: a real sqlite database with the real schema, in a temp folder that
goes away with the test.

A real database and not a stand-in, deliberately. Most of what these four steps
promise is a promise about what ends up in the tables -- that a cull takes an
individual's executions with it, that a mutant's fitness comes back NULL, that
a copy inherits every column but `is_best` -- and a fake connection would be
this suite agreeing with itself about SQL nobody ran. store.py is the only
module that speaks sqlite, so using it here keeps that true of the tests too.
"""

import os
import random
import shutil
import tempfile
import unittest

from storage import store


# A handful of chromosomes that are known-good against the grammar: root CAT,
# every binary op over two operators, every L* over one variable. Written out
# rather than drawn, so a test that names one is reading the same tree tomorrow.
VALID = (
    "CAT.L2.L1.w5.w3",
    "CAT.L3.L3.w5.w1",
    "CAT.L4.L1.w5.w1",
    "CAT.L5.L4.w2.w3",
    "CAT.L3.L5.w1.w5",
    "CAT.CAT.L5.L3.L4.w3.w3.w2",
    "CAT.L2.LIN.w2.L3.L1.w5.w1",
    "CAT.L5.CAT.w4.L4.L1.w2.w1",
    "CAT.L3.SVD.w5.L4.CAT.w3.L4.L1.w1.w3",
    "CAT.L4.SVD.w3.SVD.LIN.L1.L4.L1.L2.w3.w2.w2.w4",
)

# What selection.select() reads out of a sweep's settings when it draws its
# newcomer. The real defaults, so a test is not exercising a shape the pipeline
# never passes.
CONF = {"MAX_DEPTH": 4, "BRANCH_PROB": 0.2, "UNIQUE": True}


class SweepTestCase(unittest.TestCase):
    """A test with one empty sweep in a throwaway database.

    `self.conn` is the connection, `self.run` the sweep's id. Subclasses call
    populate() and then the step under test.
    """

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="gep-search-tests-")
        # store.connect() resolves a relative path against the repo folder, so
        # the path handed to it has to be absolute or the database lands in the
        # working tree.
        self.conn = store.connect(os.path.join(self.folder, "test.sqlite3"))
        self.run = store.create_run(self.conn, "template_code_mocked.py",
                                    label="unit test")
        self.addCleanup(self._teardown)

    def _teardown(self):
        self.conn.close()
        shutil.rmtree(self.folder, ignore_errors=True)

    # --- building a population --------------------------------------------

    def populate(self, chromosomes=None, count=None):
        """Store a population. -> the chromosomes it was given.

        Numbers run from 1, the way add_individuals hands them out.
        """
        if chromosomes is None:
            chromosomes = list(VALID[:count if count is not None else 4])
        store.add_individuals(self.conn, self.run, chromosomes)
        return list(chromosomes)

    def set_fitness(self, mapping):
        """Write individuals.fitness for `{number: fitness}`.

        The column the search sorts on, set directly: fitness is what
        calculate_fitness produces, and the three steps that read it should be
        testable without running that one first.
        """
        for number, fitness in mapping.items():
            store.set_fitness(self.conn, self.run, number, fitness)
        self.conn.commit()

    def run_individual(self, number, qualities, verdict="ok", exit_code=0):
        """Give one individual an execution and a graded transcript.

        `qualities` is one score per answer, None for an answer nobody graded.
        Returns the execution id, so a test can add a second one and check that
        only the latest counts.
        """
        row = self.individual(number)
        execution = store.add_execution(
            self.conn, row["id"], seconds=1.0, exit_code=exit_code,
            verdict=verdict, weight_seed=42,
            weights={"w%d" % slot: 0.5 for slot in range(1, 6)},
            stdout="", stderr="")
        store.add_exchanges(self.conn, execution, [
            {"question": "q%d" % position, "answer": "a%d" % position,
             "quality": quality, "reason": "because"}
            for position, quality in enumerate(qualities, 1)])
        self.conn.commit()
        return execution

    # --- reading it back ---------------------------------------------------

    def individual(self, number):
        return self.conn.execute(
            "SELECT * FROM individuals WHERE run_id = ? AND number = ?",
            (self.run, number)).fetchone()

    def population(self):
        return store.individuals(self.conn, self.run)

    def numbers(self):
        return [row["number"] for row in self.population()]

    def fitnesses(self):
        """{number: fitness} exactly as stored -- NULL stays None."""
        return {row["number"]: row["fitness"] for row in self.population()}

    def chromosomes(self):
        return {row["number"]: row["chromosome"] for row in self.population()}

    def elite(self):
        """The number of the individual flagged is_best, or None."""
        marked = [row["number"] for row in self.population() if row["is_best"]]
        self.assertLessEqual(len(marked), 1, "a sweep may hold at most one elite")
        return marked[0] if marked else None

    def execution_count(self):
        return self.conn.execute(
            "SELECT COUNT(*) AS n FROM executions e JOIN individuals i"
            " ON i.id = e.individual_id WHERE i.run_id = ?",
            (self.run,)).fetchone()["n"]

    def exchange_count(self):
        return self.conn.execute(
            "SELECT COUNT(*) AS n FROM exchanges x JOIN executions e"
            " ON e.id = x.execution_id JOIN individuals i"
            " ON i.id = e.individual_id WHERE i.run_id = ?",
            (self.run,)).fetchone()["n"]


def rng(seed=1234):
    """A generator seeded the way a sweep seeds its own: fixed, and named."""
    return random.Random(seed)
