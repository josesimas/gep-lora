"""
test_inference.py - Loaded on first use, rebuilt per deployment, gone after the TTL.
"""

import unittest

from gep_lora.service import inference


class CountingEngine:
    """An engine that remembers what was asked of it."""

    made = []

    def __init__(self, spec):
        self.loads = self.builds = self.unloads = 0
        CountingEngine.made.append(self)

    def load(self):
        self.loads += 1

    def build(self, spec):
        self.builds += 1

    def stream(self, prompt, max_new_tokens):
        yield "echo:"
        yield prompt

    def unload(self):
        self.unloads += 1


class FailingEngine(CountingEngine):
    def load(self):
        raise RuntimeError("out of memory")


def spec(engine="counting", model="base"):
    return {"engine": engine, "base_model": model, "chat_template": None,
            "chromosome": "CAT.L1.L2.w1.w2"}


class ModelCacheTests(unittest.TestCase):

    def setUp(self):
        CountingEngine.made = []
        self.cache = inference.ModelCache(
            ttl=60, sweep_every=3600,
            engines={"counting": CountingEngine, "failing": FailingEngine})

    def tearDown(self):
        self.cache.close()

    def ask(self, deployment, one=None, prompt="hi"):
        return "".join(self.cache.stream(deployment, one or spec(), prompt, 10))

    def test_the_model_loads_once_and_is_reused(self):
        self.assertEqual(self.ask(1), "echo:hi")
        self.assertEqual(self.ask(1, prompt="again"), "echo:again")
        engine, = CountingEngine.made
        self.assertEqual((engine.loads, engine.builds), (1, 1))

    def test_a_second_deployment_on_the_same_model_rebuilds_not_reloads(self):
        self.ask(1)
        self.ask(2)
        self.ask(1)
        engine, = CountingEngine.made
        self.assertEqual((engine.loads, engine.builds), (1, 3))

    def test_another_base_model_is_another_load(self):
        self.ask(1)
        self.ask(2, spec(model="other"))
        self.assertEqual(len(CountingEngine.made), 2)

    def test_an_idle_model_is_unloaded_after_the_ttl(self):
        self.ask(1)
        engine, = CountingEngine.made
        self.assertEqual(self.cache.sweep(now=0), 0)             # not idle long enough
        self.assertEqual(self.cache.sweep(now=float("inf")), 1)
        self.assertEqual(engine.unloads, 1)
        self.assertEqual(self.cache.status(), [])
        self.ask(1)                                               # loads again
        self.assertEqual(len(CountingEngine.made), 2)

    def test_a_model_in_use_is_never_unloaded(self):
        pieces = self.cache.stream(1, spec(), "hi", 10)
        next(pieces)
        self.assertEqual(self.cache.sweep(now=float("inf")), 0)
        pieces.close()
        self.assertEqual(self.cache.sweep(now=float("inf")), 1)

    def test_a_forgotten_deployment_is_rebuilt_before_it_is_served(self):
        self.ask(1)
        self.cache.forget(1)
        self.ask(1)
        self.assertEqual(CountingEngine.made[0].builds, 2)

    def test_a_failed_load_leaves_nothing_cached(self):
        with self.assertRaises(RuntimeError):
            self.ask(1, spec(engine="failing"))
        self.assertEqual(self.cache.status(), [])


class MockEngineTests(unittest.TestCase):

    def test_it_streams_a_word_at_a_time_up_to_the_cap(self):
        engine = inference.MockEngine(spec())
        engine.delay = 0
        engine.build(spec())
        pieces = list(engine.stream("hello there", 3))
        self.assertEqual(len(pieces), 3)
        self.assertTrue(pieces[1].startswith(" "))


if __name__ == "__main__":
    unittest.main()
