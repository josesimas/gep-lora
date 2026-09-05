# Features

* Stop the last generation at the elitism step
* Run the evaluation step using unsloth and local code instead of querying a remote model
* Continue a run from a database without access to anything else (settings and datasets are overriden by the contents of the database)
* Move the evaluators to the folder evaluators and put one evaluator per file (with a common file for duplicate functions)
* Get all the settings around the external services (Judge and Lora calculation) into a config.py
    * ~~Done for the judge~~ (JUDGE_*/SIMILARITY_*/HEURISTIC_*/PANEL_* in settings.py; the API key stays in $JUDGE_API_KEY)
* Add more lora types to the mix
* Refactor the code
    * Remove the excess of arguments and flags
    * Simplify the steps and move the code of each step out of main.py as much as possible
* Log to a file alongside writing to the console
* Remove redundant data from the sqlite file (trees for example)
* Create a parallel pipeline that is less batch oriented (process one chromosome only end to end)
* Create a pipeline that only loads the weights once during the run (if possible) and once during the testing
* Explore ways to avoid loading and unloading the same weights more than once
* Add parallelism to the model evaluation and processing steps 
    * ~~Done for lora processing~~
    * TBD for evaluation
* After a run is done and tested go through the answers of a specific lora combine and analyse it more deeply (list the less successful answers and try to explain why they are weak, do the opposite for the best models). The idea is to provide the user with a report with the weaknesses and strong points of this model.
* Add complexity and "time to load" as values that influence the fitness (or some fitnesses). We may want to favour a combine that loads very fast.
* Save timings for each run processing and each evaluation by adding new columns to existing table and recording the time in seconds the process took.
* Evolve the loras themselves

* ~~Order the settings in a more sane way~~
* ~~Track fitness over the generations~~
* ~~More fitness functions~~ (EVALUATOR in settings.py: llm_judge, llm_judge_reference, similarity, heuristic, panel)
* ~~Run the training set through the base model and use the results in the evaluation of the loras (send then to the judge LLM)~~
* ~~The dataset's own answers are sent to the judge~~ (EVALUATOR = "llm_judge_reference")
* ~~Save all settings and data to the run database~~
