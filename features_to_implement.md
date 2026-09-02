# Features

* Save all settings and data to the run database
* Move the evaluators to the folder evaluators and put one evaluator per file (with a common file for duplicate functions)
* ~~Order the settings in a more sane way~~
* ~~Track fitness over the generations~~
* ~~More fitness functions~~ (EVALUATOR in settings.py: llm_judge, llm_judge_reference, similarity, heuristic, panel)
* ~~Run the training set through the base model and use the results in the evaluation of the loras (send then to the judge LLM)~~
    * ~~The dataset's own answers are sent to the judge~~ (EVALUATOR = "llm_judge_reference")
* Get all the settings around the external services (Judge and Lora calculation) into a config.py
    * ~~Done for the judge~~ (JUDGE_*/SIMILARITY_*/HEURISTIC_*/PANEL_* in settings.py; the API key stays in $JUDGE_API_KEY)
* Add more lora types to the mix
* Refactor the code
    * Remove the excess of arguments and flags
    * Simplify the steps and move the code of each step out of main.py as much as possible
* Log to a file alongside writing to the console
* Remove redundant data from the sqlite file (trees for example)
* Create a parallel pipeline that is less batch oriented (process one chromosome only end to end)
* Explore ways to avoid loading and unloading the same weights more than once
* Add parallelism to the model evaluation and processing steps 
    * ~~Done for lora processing~~
    * TBD for evaluation
* Run the evaluation step using unsloth and local code instead of querying a remote model