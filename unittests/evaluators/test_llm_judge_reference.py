"""
test_llm_judge_reference.py - The judge, shown the dataset's own answer.

This evaluator is llm_judge plus a bigger prompt, so what is worth testing is
exactly the part that is its own:

  * prepare() extends llm_judge's rather than replacing it, and loads the
    references onto the bundle;
  * an eval set carrying no answers at all is fatal in prepare(), once, rather
    than one failed exchange at a time;
  * score() shows the judge the question, the reference and the answer, under
    this module's rubric;
  * an item with no reference falls back to llm_judge.score() and is graded on
    merit -- dropping it would quietly shrink the eval set for one individual
    and make its fitness incomparable with the rest;
  * a sweep's *stored* prompt still wins over the module constant, so a sweep
    created while the rubric was a setting is re-scored against the text it
    actually ran under.

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
from evaluators import common, llm_judge, llm_judge_reference


# One dataset in the shape the eval set really has: a JSON record per line
# carrying a messages list. The third record has no assistant turn, which is
# the "no reference" case score() has to fall back on.
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
    """A stand-in for common.ask_judge that records what it was asked.

    The evaluators reach the judge through exactly this function, on either
    backend, so it is the one place a test can see the prompt an evaluator
    built without a model or a socket being involved.
    """

    def __init__(self, result=(0.75, "sounds like the reference")):
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


class ReferenceEvaluatorTestCase(unittest.TestCase):
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
        return llm_judge_reference.prepare(conf or self.conf,
                                           self.pending if pending is None else pending)

    def scoring(self, result=(0.75, "sounds like the reference")):
        """Patch the judge transport. -> the JudgeCall it was replaced with."""
        judge = JudgeCall(result)
        patch = mock.patch.object(common, "ask_judge", judge)
        patch.start()
        self.addCleanup(patch.stop)
        return judge


class RegistrationTests(ReferenceEvaluatorTestCase):
    """Importing the module is the registration."""

    def test_it_is_in_the_registry_under_its_own_name(self):
        self.assertEqual(evaluators.get("llm_judge_reference").name,
                         "llm_judge_reference")

    def test_it_is_listed_for_the_evaluators_flag(self):
        self.assertIn("llm_judge_reference",
                      [name for name, _description in evaluators.available()])

    def test_it_registers_this_module_functions(self):
        entry = evaluators.get("llm_judge_reference")
        self.assertIs(entry.prepare, llm_judge_reference.prepare)
        self.assertIs(entry.score, llm_judge_reference.score)

    def test_it_wants_a_reference(self):
        self.assertTrue(evaluators.get("llm_judge_reference").wants_reference)

    def test_it_needs_a_judge_so_the_abandon_rule_applies(self):
        # Only an evaluator that asks a model has anything to save by giving
        # up on an individual that opens badly.
        self.assertTrue(evaluators.get("llm_judge_reference").needs_judge)

    def test_it_does_not_want_a_baseline(self):
        # That is llm_judge_baseline's, and it costs a base-model load.
        self.assertFalse(evaluators.get("llm_judge_reference").wants_baseline)

    def test_registering_it_did_not_displace_llm_judge(self):
        self.assertIs(evaluators.get("llm_judge").score, llm_judge.score)


class RubricTests(unittest.TestCase):
    """The prompt is this evaluator's own text, and it has a job to do."""

    PROMPT = llm_judge_reference.JUDGE_REFERENCE_SYSTEM_PROMPT

    def test_it_names_the_three_things_the_judge_is_shown(self):
        for heading in ("QUESTION", "REFERENCE ANSWER", "ANSWER"):
            self.assertIn(heading, self.PROMPT)

    def test_it_does_not_ask_for_a_copy(self):
        # A blend that reproduced the reference word for word would score well
        # and have learned nothing but that one answer.
        self.assertIn("Do not reward copying", self.PROMPT)

    def test_it_grades_on_manner(self):
        # The thing merit-only grading cannot see, and the reason this
        # evaluator exists.
        self.assertIn("Manner", self.PROMPT)

    def test_it_asks_for_the_score_before_the_reason(self):
        # A judge parsing trap already fixed once: a long reason must not
        # truncate the score away.
        self.assertLess(self.PROMPT.index("quality"), self.PROMPT.index("reason"))

    def test_it_asks_for_json_and_nothing_else(self):
        self.assertIn("Reply with JSON and nothing else", self.PROMPT)

    def test_it_is_a_different_rubric_from_the_merit_one(self):
        self.assertNotEqual(self.PROMPT, llm_judge.JUDGE_SYSTEM_PROMPT)

    def test_it_is_not_a_setting(self):
        # It lives beside the code that sends it. settings.py holds knobs; the
        # evaluators hold their own text.
        from config import settings
        self.assertFalse(hasattr(settings, "JUDGE_REFERENCE_SYSTEM_PROMPT"))


class PrepareTests(ReferenceEvaluatorTestCase):
    """prepare() is llm_judge's, plus the references."""

    def test_it_loads_the_references_from_the_eval_set(self):
        prepared = self.prepared()
        self.assertEqual(prepared.references["by_position"],
                         {1: "A raised body temperature.",
                          2: "Persistently high blood pressure."})

    def test_it_keys_the_references_by_question_too(self):
        # Matching on the question is what survives a re-run: an exchange
        # stores the question it actually asked, so it cannot be paired with
        # the wrong reference the way an index into an edited file could.
        prepared = self.prepared()
        self.assertEqual(prepared.references["by_question"]["what is a fever?"],
                         "A raised body temperature.")

    def test_a_record_with_no_assistant_turn_contributes_no_reference(self):
        prepared = self.prepared()
        self.assertNotIn(3, prepared.references["by_position"])

    def test_it_keeps_llm_judge_settings_block(self):
        # Backend, model, timeouts and retries are llm_judge's, whose prepare()
        # this one extends rather than replaces.
        prepared = self.prepared()
        self.assertEqual(prepared.settings["model"], "a-judge-model")
        self.assertEqual(prepared.settings["backend"], common.ENDPOINT)
        self.assertEqual(prepared.settings["base_url"],
                         "http://127.0.0.1:1234/v1")

    def test_the_label_is_the_judge_model(self):
        # What lands in exchanges.judge_model: what gave this score.
        self.assertEqual(self.prepared().label, "a-judge-model")

    def test_it_carries_the_sweep_settings_through(self):
        self.assertIs(self.prepared().conf, self.conf)

    def test_it_notes_how_many_references_and_where_from(self):
        notes = self.prepared().notes
        self.assertTrue(any("reference answers: 2" in note for note in notes),
                        notes)
        self.assertTrue(any(self.dataset in note for note in notes), notes)

    def test_it_keeps_the_note_llm_judge_made(self):
        notes = self.prepared().notes
        self.assertTrue(any("a-judge-model" in note and "judge:" in note
                            for note in notes), notes)

    def test_the_training_count_caps_the_references(self):
        # Same file, same order and the same cap the scripts ask under.
        prepared = self.prepared(dict(self.conf, TRAINING_COUNT=1))
        self.assertEqual(list(prepared.references["by_position"]), [1])

    def test_an_eval_set_with_no_answers_is_fatal_here(self):
        # Fatal in prepare() rather than per exchange: finding out one
        # exchange at a time would spend a whole evaluate step to say so.
        plain = write_plain(self.folder, "plain.txt",
                            ["What is a fever?", "Define hypertension."])
        with self.assertRaises(SystemExit) as caught:
            self.prepared(dict(self.conf, TRAINING_SET=plain))
        message = str(caught.exception)
        self.assertIn("llm_judge_reference", message)
        self.assertIn(plain, message)

    def test_it_does_not_reach_for_a_judge_when_nothing_needs_grading(self):
        # A sweep where every script failed is scored 0.0 without anyone being
        # asked, so an endpoint that need not be up must not be demanded.
        with mock.patch.object(common, "discover_model") as discover:
            prepared = self.prepared(dict(self.conf, JUDGE_MODEL=None),
                                     pending=[item(1, "q", answer="   ")])
        discover.assert_not_called()
        self.assertTrue(any("not contacted" in note for note in prepared.notes))


class ScoreTests(ReferenceEvaluatorTestCase):
    """score() is where the reference actually reaches the judge."""

    def test_it_returns_what_the_judge_said(self):
        self.scoring((0.6, "close enough"))
        prepared = self.prepared()
        self.assertEqual(llm_judge_reference.score(self.pending[0], prepared),
                         (0.6, "close enough"))

    def test_it_shows_the_judge_the_reference(self):
        judge = self.scoring()
        prepared = self.prepared()
        llm_judge_reference.score(item(1, "What is a fever?", "Warm."), prepared)
        self.assertIn("REFERENCE ANSWER:\nA raised body temperature.",
                      judge.content)

    def test_it_shows_the_question_the_reference_and_the_answer_in_order(self):
        judge = self.scoring()
        prepared = self.prepared()
        llm_judge_reference.score(item(1, "What is a fever?", "Warm."), prepared)
        self.assertEqual(
            judge.content,
            "QUESTION:\nWhat is a fever?\n\n"
            "REFERENCE ANSWER:\nA raised body temperature.\n\n"
            "ANSWER:\nWarm.")

    def test_it_sends_this_module_rubric(self):
        judge = self.scoring()
        prepared = self.prepared()
        llm_judge_reference.score(self.pending[0], prepared)
        self.assertEqual(judge.prompt,
                         llm_judge_reference.JUDGE_REFERENCE_SYSTEM_PROMPT)

    def test_it_sends_the_resolved_judge_settings(self):
        judge = self.scoring()
        prepared = self.prepared()
        llm_judge_reference.score(self.pending[0], prepared)
        self.assertIs(judge.calls[-1]["settings"], prepared.settings)

    def test_a_stored_prompt_wins_over_the_module_constant(self):
        # A sweep created while the rubric was a setting is re-scored against
        # the prompt it actually ran with.
        judge = self.scoring()
        prepared = self.prepared(
            dict(self.conf, JUDGE_REFERENCE_SYSTEM_PROMPT="grade it my way"))
        llm_judge_reference.score(self.pending[0], prepared)
        self.assertEqual(judge.prompt, "grade it my way")

    def test_a_blank_stored_prompt_falls_back_to_the_constant(self):
        judge = self.scoring()
        prepared = self.prepared(dict(self.conf, JUDGE_REFERENCE_SYSTEM_PROMPT=""))
        llm_judge_reference.score(self.pending[0], prepared)
        self.assertEqual(judge.prompt,
                         llm_judge_reference.JUDGE_REFERENCE_SYSTEM_PROMPT)

    def test_the_reference_is_matched_on_the_question(self):
        # Position 2's question, arriving at some other position, must still
        # be graded against its own reference.
        judge = self.scoring()
        prepared = self.prepared()
        llm_judge_reference.score(item(99, "Define hypertension."), prepared)
        self.assertIn("Persistently high blood pressure.", judge.content)

    def test_the_question_match_ignores_case_and_spacing(self):
        judge = self.scoring()
        prepared = self.prepared()
        llm_judge_reference.score(item(99, "  WHAT   IS a Fever?  "), prepared)
        self.assertIn("A raised body temperature.", judge.content)

    def test_position_is_the_fallback_when_the_question_does_not_match(self):
        judge = self.scoring()
        prepared = self.prepared()
        llm_judge_reference.score(item(2, "a question nobody stored"), prepared)
        self.assertIn("Persistently high blood pressure.", judge.content)


class NoReferenceTests(ReferenceEvaluatorTestCase):
    """An item the eval set has no answer for is graded on merit, not dropped."""

    def test_it_falls_back_to_the_merit_rubric(self):
        judge = self.scoring()
        prepared = self.prepared()
        llm_judge_reference.score(item(3, "Name a symptom."), prepared)
        self.assertEqual(judge.prompt, llm_judge.JUDGE_SYSTEM_PROMPT)

    def test_the_fallback_shows_no_reference_section(self):
        judge = self.scoring()
        prepared = self.prepared()
        llm_judge_reference.score(item(3, "Name a symptom.", "A cough."), prepared)
        self.assertEqual(judge.content,
                         "QUESTION:\nName a symptom.\n\nANSWER:\nA cough.")

    def test_the_fallback_still_produces_a_score(self):
        # Dropping the answer would quietly shrink the eval set for that
        # individual and make its fitness incomparable with the rest.
        self.scoring((0.4, "graded on merit"))
        prepared = self.prepared()
        self.assertEqual(
            llm_judge_reference.score(item(3, "Name a symptom."), prepared),
            (0.4, "graded on merit"))

    def test_an_empty_reference_falls_back_to_merit(self):
        # An assistant turn that is present but blank is no reference at all,
        # so that one item is graded on merit while its neighbours are not.
        dataset = write_dataset(self.folder, "blank_ref.json",
                                [("Real question.", "A real answer."),
                                 ("Blank one.", "")])
        judge = self.scoring()
        prepared = self.prepared(dict(self.conf, TRAINING_SET=dataset))
        self.assertEqual(list(prepared.references["by_position"]), [1])

        llm_judge_reference.score(item(2, "Blank one."), prepared)
        self.assertEqual(judge.prompt, llm_judge.JUDGE_SYSTEM_PROMPT)

        llm_judge_reference.score(item(1, "Real question."), prepared)
        self.assertEqual(judge.prompt,
                         llm_judge_reference.JUDGE_REFERENCE_SYSTEM_PROMPT)

    def test_an_eval_set_of_only_blank_answers_is_refused(self):
        # Not a fallback case: an eval set carrying no usable answer anywhere
        # cannot be graded against references at all, and prepare() says so
        # once rather than one exchange at a time.
        dataset = write_dataset(self.folder, "all_blank.json",
                                [("Only question.", "")])
        judge = self.scoring()
        with self.assertRaises(SystemExit):
            self.prepared(dict(self.conf, TRAINING_SET=dataset))
        self.assertEqual(judge.calls, [], "no answer should have been graded")

    def test_a_stored_merit_prompt_reaches_the_fallback(self):
        judge = self.scoring()
        prepared = self.prepared(
            dict(self.conf, JUDGE_SYSTEM_PROMPT="merit, my way"))
        llm_judge_reference.score(item(3, "Name a symptom."), prepared)
        self.assertEqual(judge.prompt, "merit, my way")


class FailureTests(ReferenceEvaluatorTestCase):
    """A failed grading call fails that one answer and no more."""

    def test_a_judge_failure_propagates_out_of_score(self):
        # start_run.py counts it and moves on, which is what makes a
        # half-scored sweep resumable.
        patch = mock.patch.object(
            common, "ask_judge",
            mock.Mock(side_effect=RuntimeError("judge unreachable")))
        patch.start()
        self.addCleanup(patch.stop)
        prepared = self.prepared()
        with self.assertRaises(RuntimeError):
            llm_judge_reference.score(self.pending[0], prepared)

    def test_an_unparseable_reply_propagates_too(self):
        patch = mock.patch.object(
            common, "ask_judge",
            mock.Mock(side_effect=ValueError("no quality score")))
        patch.start()
        self.addCleanup(patch.stop)
        prepared = self.prepared()
        with self.assertRaises(ValueError):
            llm_judge_reference.score(self.pending[0], prepared)


if __name__ == "__main__":
    unittest.main()
