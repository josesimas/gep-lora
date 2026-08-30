# Features

* ~~Track fitness over the generations~~
* More fitness functions
* Run the training set through the base model and use the results in the evaluation of the loras (send then to the judge LLM)
* Get all the settings around the external services (Judge and Lora calculation) into a config.py
* Add more lora types to the mix
* Refactor the code
    * Remove the excess of arguments and flags
    * Simplify the steps and move the code of each step out of main.py as much as possible
* Log to a file alongside writing to the console
* Remove redundant data from the sqlite file (trees for example)
* Create a parallel pipeline that is less batch oriented (process one chromosome only end to end)
* Explore ways to avoid loading and unloading the same weights more than once
* Add parallelism to the model evaluation and processing steps 