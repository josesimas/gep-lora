# Features
* Get all the settings around the external services (Judge and Lora calculation) into a config.py
    * ~~Done for the judge~~ (JUDGE_*/SIMILARITY_*/HEURISTIC_*/PANEL_* in settings.py; the API key stays in $JUDGE_API_KEY)
* Add more lora types to the mix
* Refactor the code
    * Remove the excess of arguments and flags
    * Simplify the steps and move the code of each step out of start_run.py as much as possible
* Log to a file alongside writing to the console
* Remove redundant data from the sqlite file (trees for example)
* Create a parallel pipeline that is less batch oriented (process one chromosome only end to end)
* ~~Create a pipeline that only loads the weights once during the run (if possible) and once during the testing~~ (TEMPLATE = "template_remote_code.py": the scripts become clients of blends/lora_server.py, which holds the base model open; the testing pass starts a pool the same way. LORA_SERVER_* in settings.py)
* Explore ways to avoid loading and unloading the same weights more than once
    * ~~The base model~~ (lora_server.py loads it once per server per step instead of once per individual)
    * TBD for the adapters themselves, which are still attached and deleted per blend
* Add parallelism to the model evaluation and processing steps 
    * ~~Done for lora processing~~
    * TBD for evaluation
* After a run is done and tested go through the answers of a specific lora combine and analyse it more deeply (list the less successful answers and try to explain why they are weak, do the opposite for the best models). The idea is to provide the user with a report with the weaknesses and strong points of this model.
* Add complexity and "time to load" as values that influence the fitness (or some fitnesses). We may want to favour a combine that loads very fast.
* Evolve the loras themselves
* Add a random constant mutation (only changes the values of the weights)
* Train Loras on random subsets of the training set
* Change the fitness values to more granular values (maybe 20 options)
* Add an escalation feature where later generations are scored by better models (cloud models instead local)

* ~~Save timings for each run processing and each evaluation by adding new columns to existing table and recording the time in seconds the process took.~~
* ~~Move the evaluators to the folder evaluators and put one evaluator per file (with a common file for duplicate functions)~~
* ~~Stop the last generation at the elitism step~~
* ~~Run the evaluation step using unsloth and local code instead of querying a remote model~~
* ~~Continue a run from a database without access to anything else (settings and datasets are overriden by the contents of the database)~~
* ~~Order the settings in a more sane way~~
* ~~Track fitness over the generations~~
* ~~More fitness functions~~ (EVALUATOR in settings.py: llm_judge, llm_judge_reference, similarity, heuristic, panel)
* ~~Run the training set through the base model and use the results in the evaluation of the loras (send then to the judge LLM)~~
* ~~The dataset's own answers are sent to the judge~~ (EVALUATOR = "llm_judge_reference")
* ~~Save all settings and data to the run database~~
