"""
test_llm_judge_answers.py - The judge, shown two answers and nothing else.

This evaluator is llm_judge's prepare() plus a rubric of its own, so what is
worth testing is exactly the part that is its own:

  * prepare() extends llm_judge's rather than replacing it, and loads the
    references onto the bundle;
  * an eval set carrying no answers at all is fatal in prepare(), once, rather
    than one failed exchange at a time;
  * score() sends the reference and the answer under this module's rubric --
    and, the whole point of the evaluator, **not the question**, even though
    the question is what found the reference;
  * an item with no reference fails that one exchange rather than falling back
    to llm_judge.score(), which would send the question and grade a different
    quantity from the answers around it.

Nothing here contacts a judge. `common.ask_judge` is the seam every judging
evaluator goes through, so patching it both keeps the tests off the network and
lets them read back the exact prompt and content this evaluator sent -- which
is the thing under test. The transport below it (retries, HTTP codes, reply
parsing) is common.py's and is not re-tested here.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import evaluators
from evaluators import common, llm_judge, llm_judge_answers


# One dataset in the shape the eval set really has: a JSON record per line
# carrying a messages list. The third record has no assistant turn, which is
# the "no reference" case -- and the case this evaluator refuses rather than
# falls back on.
RECORDS = [
    ("What is a fever?", "A raised body temperature."),
    ("Define hypertension.", "Persistently high blood pressure."),
    ("Name a symptom.", None),
]


def write_dataset(folder, name, records):
    """A JSON-lines eval file. -> its absolute path."""
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as handle:
        for question, reference in records:
            messages = [{"role": "user", "content": question}]
            if reference is not None:
                messages.append({"role": "assistant", "content": reference})
            handle.write(json.dumps({"messages": messages}) + "\n")
    return path


def write_plain(folder, name, questions):
    """A plain one-prompt-per-line eval file -- the shape that has no answers."""
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(questions) + "\n")
    return path


def item(position, question, answer="an answer"):
    """One pending exchange, as the evaluate step hands it to score()."""
    return {"position": position, "question": question, "answer": answer}


class JudgeCall:
    """A stand-in for common.ask_judge that records what it was asked."""

    def __init__(self, result=(0.75, "stands in for the reference")):
        self.result = result
        self.calls = []

    def __call__(self, system_prompt, user_content, settings):
        self.calls.append({"prompt": system_prompt, "content": user_content,
                           "settings": settings})
        return self.result

    @property
    def prompt(self):
        return self.calls[-1]["prompt"]

    @property
    def content(self):
        return self.calls[-1]["content"]


class AnswersEvaluatorTestCase(unittest.TestCase):
    """A sweep's settings pointed at a temp eval set, and no judge anywhere."""

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="gep-evaluator-tests-")
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        self.dataset = write_dataset(self.folder, "eval.json", RECORDS)
        # JUDGE_MODEL is named so that prepare() never reaches for
        # discover_model(), which would try to open a socket.
        self.conf = {"TRAINING_SET": self.dataset,
                     "TRAINING_COUNT": None,
                     "JUDGE_BACKEND": "endpoint",
                     "JUDGE_BASE_URL": "http://127.0.0.1:1234/v1",
                     "JUDGE_MODEL": "a-judge-model"}
        self.pending = [item(1, "What is a fever?"),
                        item(2, "Define hypertension."),
                        item(3, "Name a symptom.")]

    def prepared(self, conf=None, pending=None):
        return llm_judge_answers.prepare(conf or self.conf,
                                         self.pending if pending is None else pending)

    def scoring(self, result=(0.75, "stands in for the reference")):
        """Patch the judge transport. -> the JudgeCall it was replaced with."""
        judge = JudgeCall(result)
        patch = mock.patch.object(common, "ask_judge", judge)
        patch.start()
        self.addCleanup(patch.stop)
        return judge


class RegistrationTests(AnswersEvaluatorTestCase):
    """Importing the module is the registration."""

    def test_it_is_in_the_registry_under_its_own_name(self):
        self.assertEqual(evaluators.get("llm_judge_answers").name,
                         "llm_judge_answers")

    def test_it_is_listed_for_the_evaluators_flag(self):
        self.assertIn("llm_judge_answers",
                      [name for name, _description in evaluators.available()])

    def test_it_registers_this_module_functions(self):
        entry = evaluators.get("llm_judge_answers")
        self.assertIs(entry.prepare, llm_judge_answers.prepare)
        self.assertIs(entry.score, llm_judge_answers.score)

    def test_it_wants_a_reference(self):
        self.assertTrue(evaluators.get("llm_judge_answers").wants_reference)

    def test_it_needs_a_judge_so_the_abandon_rule_applies(self):
        self.assertTrue(evaluators.get("llm_judge_answers").needs_judge)

    def test_it_does_not_want_a_baseline(self):
        self.assertFalse(evaluators.get("llm_judge_answers").wants_baseline)

    def test_registering_it_did_not_displace_the_others(self):
        self.assertIs(evaluators.get("llm_judge").score, llm_judge.score)


class RubricTests(unittest.TestCase):
    """The prompt is this evaluator's own text, and it has a job to do."""

    PROMPT = llm_judge_answers.JUDGE_ANSWERS_SYSTEM_PROMPT

    def test_it_names_the_two_things_the_judge_is_shown(self):
        for heading in ("REFERENCE ANSWER", "ANSWER"):
            self.assertIn(heading, self.PROMPT)

    def test_it_says_the_question_is_withheld(self):
        # A judge that has not been told will look for the question and
        # complain about its absence instead of grading. Matched on the
        # unwrapped text: the rubric is hard-wrapped, and a line break falling
        # inside the phrase is not a change of meaning.
        self.assertIn("NOT shown the question", " ".join(self.PROMPT.split()))

    def test_it_does_not_ask_for_a_copy(self):
        self.assertIn("Do not reward copying", self.PROMPT)

    def test_it_grades_on_manner(self):
        self.assertIn("Manner", self.PROMPT)

    def test_it_asks_for_the_score_before_the_reason(self):
        # A judge parsing trap already fixed once: a long reason must not
        # truncate the score away.
        self.assertLess(self.PROMPT.index("quality"), self.PROMPT.index("reason"))

    def test_it_asks_for_json_and_nothing_else(self):
        self.assertIn("Reply with JSON and nothing else", self.PROMPT)

    def test_it_is_a_rubric_of_its_own(self):
        from evaluators import llm_judge_reference
        self.assertNotEqual(self.PROMPT, llm_judge.JUDGE_SYSTEM_PROMPT)
        self.assertNotEqual(self.PROMPT,
                            llm_judge_reference.JUDGE_REFERENCE_SYSTEM_PROMPT)

    def test_it_is_not_a_setting(self):
        # settings.py holds knobs; the evaluators hold their own text.
        from config import settings
        self.assertFalse(hasattr(settings, "JUDGE_ANSWERS_SYSTEM_PROMPT"))


class PrepareTests(AnswersEvaluatorTestCase):
    """prepare() is llm_judge's, plus the references."""

    def test_it_carries_the_judge_settings_llm_judge_built(self):
        prepared = self.prepared()
        self.assertEqual(prepared.settings["model"], "a-judge-model")
        self.assertEqual(prepared.settings["backend"], common.ENDPOINT)

    def test_it_labels_the_scores_with_the_judge_model(self):
        self.assertEqual(self.prepared().label, "a-judge-model")

    def test_it_loads_the_references_onto_the_bundle(self):
        prepared = self.prepared()
        self.assertEqual(prepared.references["by_question"]["what is a fever?"],
                         "A raised body temperature.")

    def test_it_says_in_a_note_that_the_question_is_not_sent(self):
        # The one line a reader of the step's output has to tell this
        # evaluator apart from llm_judge_reference.
        self.assertTrue(any("the question is not sent" in note
                            for note in self.prepared().notes))

    def test_an_eval_set_with_no_answers_is_fatal_here_and_names_this_evaluator(self):
        # Once, rather than one failed exchange at a time: an evaluator that
        # compares against the dataset's answers cannot score such a file at
        # all, and finding out one exchange at a time would spend a whole step.
        conf = dict(self.conf,
                    TRAINING_SET=write_plain(self.folder, "plain.txt",
                                             ["What is a fever?"]))
        with self.assertRaises(SystemExit) as raised:
            llm_judge_answers.prepare(conf, self.pending)
        self.assertIn("llm_judge_answers", str(raised.exception))

    def test_it_asks_no_judge_while_preparing(self):
        judge = self.scoring()
        self.prepared()
        self.assertEqual(judge.calls, [])


class ScoreTests(AnswersEvaluatorTestCase):
    """What actually reaches the judge -- two answers, and no question."""

    def test_it_returns_what_the_judge_said(self):
        self.scoring((0.4, "half the manner"))
        self.assertEqual(llm_judge_answers.score(self.pending[0], self.prepared()),
                         (0.4, "half the manner"))

    def test_it_sends_the_reference_and_the_answer(self):
        judge = self.scoring()
        llm_judge_answers.score(item(1, "What is a fever?", "Warm."), self.prepared())
        self.assertIn("A raised body temperature.", judge.content)
        self.assertIn("Warm.", judge.content)

    def test_it_does_not_send_the_question(self):
        # The whole evaluator. With the question in front of it a judge grades
        # merit as well as manner, and this one is meant to measure agreement
        # with the reference and nothing else.
        judge = self.scoring()
        llm_judge_answers.score(item(1, "What is a fever?", "Warm."), self.prepared())
        self.assertNotIn("What is a fever?", judge.content)
        self.assertNotIn("QUESTION", judge.content)

    def test_it_sends_this_modules_rubric(self):
        judge = self.scoring()
        llm_judge_answers.score(self.pending[0], self.prepared())
        self.assertEqual(judge.prompt, llm_judge_answers.JUDGE_ANSWERS_SYSTEM_PROMPT)

    def test_it_sends_the_judge_settings_the_bundle_carries(self):
        judge = self.scoring()
        prepared = self.prepared()
        llm_judge_answers.score(self.pending[0], prepared)
        self.assertIs(judge.calls[-1]["settings"], prepared.settings)

    def test_the_question_still_chooses_the_reference(self):
        # It is used to find the reference and then dropped, which is what
        # stops an answer being paired with somebody else's reference.
        judge = self.scoring()
        llm_judge_answers.score(item(99, "Define hypertension."), self.prepared())
        self.assertIn("Persistently high blood pressure.", judge.content)

    def test_the_question_is_matched_however_it_was_spaced(self):
        judge = self.scoring()
        llm_judge_answers.score(item(99, "  WHAT   IS a Fever?  "), self.prepared())
        self.assertIn("A raised body temperature.", judge.content)

    def test_an_unknown_question_falls_back_to_the_position(self):
        judge = self.scoring()
        llm_judge_answers.score(item(2, "a question nobody stored"), self.prepared())
        self.assertIn("Persistently high blood pressure.", judge.content)

    def test_an_item_with_no_reference_fails_that_exchange(self):
        # Unlike llm_judge_reference, which grades it on merit: the fallback
        # would send the question and score a different quantity from the
        # answers around it. One failed exchange is the contract.
        judge = self.scoring()
        with self.assertRaises(ValueError) as raised:
            llm_judge_answers.score(item(3, "Name a symptom."), self.prepared())
        self.assertIn("never shown the question", str(raised.exception))
        self.assertEqual(judge.calls, [])

    def test_a_failed_exchange_does_not_stop_the_next_one(self):
        judge = self.scoring()
        prepared = self.prepared()
        with self.assertRaises(ValueError):
            llm_judge_answers.score(item(3, "Name a symptom."), prepared)
        llm_judge_answers.score(item(1, "What is a fever?"), prepared)
        self.assertEqual(len(judge.calls), 1)

    def test_a_judge_that_cannot_be_reached_fails_the_exchange(self):
        # RuntimeError out of the transport is the evaluate step's signal to
        # count this answer and move on, and it must not be swallowed here.
        with mock.patch.object(common, "ask_judge",
                               side_effect=RuntimeError("judge unreachable")):
            with self.assertRaises(RuntimeError):
                llm_judge_answers.score(self.pending[0], self.prepared())


if __name__ == "__main__":
    unittest.main()
