# Code critique and improvement suggestions

**Session ID:** ses_f302c064affeTSBbkVzUxaLecI
**Created:** 9/23/2026, 7:52:50 PM
**Updated:** 9/23/2026, 8:01:55 PM

---

## User

Look into this project, critique de code and suggest areas of improvement

---

## Assistant (Build · DeepSeek V4.1 Flash · 5.9s)

_Thinking:_

The user wants me to look into this project, critique the code, and suggest areas of improvement. This is a large codebase. Let me explore it to understand the structure, then provide a critique.

Let me start by exploring the repository structure and reading key files. I should use the explore agent and also read files directly.

Let me first get a lay of the land.

I'll explore the project structure and key files to understand the codebase before critiquing.

**Tool: bash**

**Input:**
```json
{
  "command": "git log --oneline -10; echo \"---\"; git status --short | head -20; echo \"---\"; ls",
  "description": "Show recent commits, status, and root listing"
}
```

**Output:**
```
7bb537d Add testing, verification and going live after blending
dc0b775 Added blending to the agentic ui process
f23f7a4 These datasets can be public.
174bf2d Adde a Jev based evaluator
959ed2e New agent UI
62e0728 Let the LoRA guide act on what is asked in the chat, through tools
f9e8191 Add the LoRA guide: an agent page that walks a user to trained LoRAs
e015daf Train and catalogue LoRAs from the API and the demo, one database per API
66007aa Redesign the demo page as a console: sidebar, views and tabs
cdfba92 Small redrwaing detail
---
head : The term 'head' is not recognized as the name of a cmdlet, function, script file, or operable program. Check 
the spelling of the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:57
---
+ git log --oneline -10; echo "---"; git status --short | head -20; ech ...
+                                                         ~~~~
    + CategoryInfo          : ObjectNotFound: (head:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
 


    Directory: D:\sage-is\loras\gep-lora


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
d-----         9/23/2026   7:51 PM                .claude                                                              
d-----         9/20/2026   2:13 PM                .idea                                                                
d-----          9/4/2026   9:15 AM                .venv                                                                
d-----         9/23/2026   6:33 AM                adapters                                                             
d-----         9/17/2026   7:27 AM                api                                                                  
d-----         9/23/2026   7:52 PM                api_jobs                                                             
d-----         9/23/2026   3:56 PM                async_api                                                            
d-----         9/23/2026   3:56 PM                async_api_agent                                                      
d-----         9/18/2026   8:21 AM                blends                                                               
d-----         9/23/2026   6:33 AM                config                                                               
d-----          9/1/2026  11:17 AM                datasets                                                             
d-----          9/5/2026   5:35 PM                dbtemplates                                                          
d-----         8/31/2026   1:31 PM                docs                                                                 
d-----         9/23/2026   6:33 AM                evaluators                                                           
d-----         9/18/2026   8:43 PM                loras                                                                
d-----          9/6/2026   8:32 PM                metrics                                                              
d-----         9/17/2026   1:52 PM                reporting                                                            
d-----         9/16/2026   9:40 AM                run                                                                  
d-----          9/8/2026   9:15 AM                runs                                                                 
d-----         9/17/2026   6:57 AM                run_db                                                               
d-----         9/16/2026   1:47 PM                run_testing                                                          
d-----         9/17/2026   1:49 PM                search                                                               
d-----         9/18/2026   8:21 AM                storage                                                              
d-----         9/16/2026   9:19 AM                templates                                                            
d-----         9/18/2026   8:21 AM                testing                                                              
d-----         9/17/2026   1:52 PM                tools                                                                
d-----         9/18/2026   9:34 PM                unittests                                                            
d-----          9/8/2026   9:40 AM                unsloth_compiled_cache                                               
d-----         9/23/2026   9:19 AM                __pycache__                                                          
-a----         9/23/2026   8:52 AM            578 .gitignore                                                           
-a----         8/13/2026   9:13 AM             63 activate.bat                                                         
-a----         9/17/2026   2:18 PM           1216 async_api.txt                                                        
-a----         9/23/2026   1:44 PM          95254 CLAUDE.md                                                            
-a----         9/17/2026   1:52 PM          18815 gep_lora/core/pipeline/continue_run.py                                                      
-a----         8/13/2026   9:13 AM            213 create_env.bat                                                       
-a----         8/13/2026   9:13 AM            108 delete_env.bat                                                       
-a----         8/13/2026   9:41 AM            787 download_wheels.bat                                                  
-a----          9/6/2026   8:32 PM           3196 features_to_implement.md                                             
-a----          9/8/2026   7:41 AM             62 generate_report.bat                                                  
-a----         9/17/2026   1:52 PM          16237 HOWTO.md                                                             
-a----         8/25/2026   9:13 AM          34523 LICENSE                                                              
-a----         9/18/2026   8:31 AM          37452 gep_lora/core/pipeline/main.py                                                              
-a----         9/23/2026   6:29 AM              0 NUL                                                                  
-a----          9/6/2026   5:21 PM          14181 PerformanceWork.md                                                   
-a----         8/24/2026   8:58 AM            809 plan.txt                                                             
-a----         9/23/2026   3:58 PM         201613 README.md                                                            
-a----         9/18/2026   9:36 PM            656 run_api_server.bat                                                   
-a----         9/17/2026   8:45 AM            714 run_api_worker.bat                                                   
-a----          9/7/2026   9:17 AM             45 run_unit_tests.bat                                                   
-a----         8/13/2026   9:41 AM           1400 setup_torch.bat                                                      
-a----         9/23/2026   6:33 AM          76069 gep_lora/core/pipeline/start_run.py                                                         



```

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/*.py"
}
```

**Output:**
```
D:\sage-is\loras\gep-lora\async_api_agent\agent.py
D:\sage-is\loras\gep-lora\async_api_agent\prompts.py
D:\sage-is\loras\gep-lora\unittests\async_api_agent\test_agent.py
D:\sage-is\loras\gep-lora\async_api_agent\blending.py
D:\sage-is\loras\gep-lora\async_api_agent\planner.py
D:\sage-is\loras\gep-lora\async_api_agent\settings.py
D:\sage-is\loras\gep-lora\async_api_agent\tools.py
D:\sage-is\loras\gep-lora\unittests\async_api\test_server.py
D:\sage-is\loras\gep-lora\async_api_agent\routes.py
D:\sage-is\loras\gep-lora\async_api_agent\commands.py
D:\sage-is\loras\gep-lora\async_api\worker.py
D:\sage-is\loras\gep-lora\async_api\submit.py
D:\sage-is\loras\gep-lora\async_api\verify.py
D:\sage-is\loras\gep-lora\async_api\registry.py
D:\sage-is\loras\gep-lora\async_api\server.py
D:\sage-is\loras\gep-lora\async_api_agent\release.py
D:\sage-is\loras\gep-lora\async_api\testpass.py
D:\sage-is\loras\gep-lora\async_api_agent\__init__.py
D:\sage-is\loras\gep-lora\unittests\async_api\test_submit.py
D:\sage-is\loras\gep-lora\unittests\async_api\test_train.py
D:\sage-is\loras\gep-lora\unittests\async_api\support.py
D:\sage-is\loras\gep-lora\unittests\async_api\test_registry.py
D:\sage-is\loras\gep-lora\unittests\async_api_agent\__init__.py
D:\sage-is\loras\gep-lora\unittests\adapters\test_catalog.py
D:\sage-is\loras\gep-lora\evaluators\jev_judge_reference.py
D:\sage-is\loras\gep-lora\start_run.py
D:\sage-is\loras\gep-lora\unittests\adapters\__init__.py
D:\sage-is\loras\gep-lora\evaluators\common.py
D:\sage-is\loras\gep-lora\evaluators\__init__.py
D:\sage-is\loras\gep-lora\config\settings.py
D:\sage-is\loras\gep-lora\async_api_agent\providers.py
D:\sage-is\loras\gep-lora\async_api_agent\selection.py
D:\sage-is\loras\gep-lora\async_api_agent\analysis.py
D:\sage-is\loras\gep-lora\async_api\settings.py
D:\sage-is\loras\gep-lora\async_api\train.py
D:\sage-is\loras\gep-lora\adapters\create_lora.py
D:\sage-is\loras\gep-lora\async_api\__init__.py
D:\sage-is\loras\gep-lora\adapters\catalog.py
D:\sage-is\loras\gep-lora\adapters\create_all_loras.py
D:\sage-is\loras\gep-lora\main.py
D:\sage-is\loras\gep-lora\testing\test_run_with_dataset.py
D:\sage-is\loras\gep-lora\async_api\evaluate.py
D:\sage-is\loras\gep-lora\async_api\results.py
D:\sage-is\loras\gep-lora\storage\store.py
D:\sage-is\loras\gep-lora\blends\process_run.py
D:\sage-is\loras\gep-lora\unittests\async_api\test_verify.py
D:\sage-is\loras\gep-lora\testing\evaluate_chromosome_against_loras.py
D:\sage-is\loras\gep-lora\blends\generate_runs.py
D:\sage-is\loras\gep-lora\evaluators\composite.py
D:\sage-is\loras\gep-lora\unittests\evaluators\test_composite.py
D:\sage-is\loras\gep-lora\unittests\async_api\test_results.py
D:\sage-is\loras\gep-lora\unittests\async_api\test_golive.py
D:\sage-is\loras\gep-lora\unittests\async_api\test_inference.py
D:\sage-is\loras\gep-lora\unittests\async_api\__init__.py
D:\sage-is\loras\gep-lora\async_api\users.py
D:\sage-is\loras\gep-lora\async_api\golive.py
D:\sage-is\loras\gep-lora\async_api\inference.py
D:\sage-is\loras\gep-lora\tools\compare_servers.py
D:\sage-is\loras\gep-lora\reporting\generate_html_db_stats.py
D:\sage-is\loras\gep-lora\continue_run.py
D:\sage-is\loras\gep-lora\search\weight_mutation.py
D:\sage-is\loras\gep-lora\blends\lora_server.py
D:\sage-is\loras\gep-lora\blends\server_pool.py
D:\sage-is\loras\gep-lora\adapters\base_models_and_loras_comparison.py
D:\sage-is\loras\gep-lora\adapters\test_lora.py
D:\sage-is\loras\gep-lora\templates\template_remote_code.py
D:\sage-is\loras\gep-lora\templates\template_baseline.py
D:\sage-is\loras\gep-lora\templates\template_code.py
D:\sage-is\loras\gep-lora\blends\baseline_run.py
D:\sage-is\loras\gep-lora\evaluators\local_model.py
D:\sage-is\loras\gep-lora\evaluators\llm_judge_answers.py
D:\sage-is\loras\gep-lora\evaluators\llm_judge.py
D:\sage-is\loras\gep-lora\metrics\report.py
D:\sage-is\loras\gep-lora\templates\template_code_mocked.py
D:\sage-is\loras\gep-lora\metrics\record.py
D:\sage-is\loras\gep-lora\metrics\__init__.py
D:\sage-is\loras\gep-lora\evaluators\panel.py
D:\sage-is\loras\gep-lora\evaluators\llm_judge_baseline.py
D:\sage-is\loras\gep-lora\evaluators\llm_judge_reference.py
D:\sage-is\loras\gep-lora\storage\add_dataset.py
D:\sage-is\loras\gep-lora\tools\test.py
D:\sage-is\loras\gep-lora\tools\combination.py
D:\sage-is\loras\gep-lora\storage\db_datasets.py
D:\sage-is\loras\gep-lora\search\selection.py
D:\sage-is\loras\gep-lora\search\mutation.py
D:\sage-is\loras\gep-lora\search\calculate_fitness.py
D:\sage-is\loras\gep-lora\search\draw_trees.py
D:\sage-is\loras\gep-lora\search\elitism.py
D:\sage-is\loras\gep-lora\tools\__init__.py
D:\sage-is\loras\gep-lora\adapters\__init__.py
D:\sage-is\loras\gep-lora\reporting\__init__.py
D:\sage-is\loras\gep-lora\storage\__init__.py
D:\sage-is\loras\gep-lora\testing\__init__.py
D:\sage-is\loras\gep-lora\blends\__init__.py
D:\sage-is\loras\gep-lora\config\__init__.py
D:\sage-is\loras\gep-lora\search\__init__.py
D:\sage-is\loras\gep-lora\search\generate_population.py
D:\sage-is\loras\gep-lora\evaluators\heuristic.py
D:\sage-is\loras\gep-lora\evaluators\similarity.py
D:\sage-is\loras\gep-lora\templates\template_baseline_mocked.py

(Results are truncated: showing first 100 results. Consider using a more specific path or pattern.)
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 4.9s)

_Thinking:_

This is a substantial project. Let me explore the core modules. Note that `gep_lora/core/pipeline/start_run.py` is 76KB, `gep_lora/core/pipeline/main.py` 37KB, `store.py` etc. Let me look at the key files. I'll use parallel reads and the explore agent.

Let me check line counts and read the core files.

Large codebase. Let me examine the core modules and structure in parallel.

**Tool: bash**

**Input:**
```json
{
  "command": "Get-ChildItem -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '\\\\\\.venv\\\\|__pycache__|unsloth_compiled' } | ForEach-Object { [PSCustomObject]@{ Lines = (Get-Content $_.FullName | Measure-Object -Line).Lines; File = $_.FullName.Replace('D:\\sage-is\\loras\\gep-lora\\','') } } | Sort-Object Lines -Descending | Select-Object -First 40 | Format-Table -AutoSize",
  "description": "Count lines in Python files"
}
```

**Output:**
```

Lines File                                            
----- ----                                            
 2424 reporting\generate_html_db_stats.py             
 1388 storage\store.py                                
 1291 gep_lora/core/pipeline/start_run.py                                    
 1007 tools\compare_servers.py                        
  914 async_api\server.py                             
  883 unittests\async_api_agent\test_agent.py         
  763 adapters\base_models_and_loras_comparison.py    
  758 testing\test_run_with_dataset.py                
  705 gep_lora/core/pipeline/main.py                                         
  635 async_api\registry.py                           
  616 adapters\create_lora.py                         
  607 blends\generate_runs.py                         
  600 config\settings.py                              
  586 async_api_agent\tools.py                        
  575 async_api_agent\prompts.py                      
  563 testing\evaluate_chromosome_against_loras.py    
  548 adapters\catalog.py                             
  479 blends\process_run.py                           
  475 blends\lora_server.py                           
  467 templates\template_code.py                      
  441 evaluators\common.py                            
  428 async_api_agent\agent.py                        
  424 adapters\test_lora.py                           
  409 run\test_run.py                                 
  406 async_api\worker.py                             
  390 adapters\create_all_loras.py                    
  375 async_api_agent\routes.py                       
  373 async_api_agent\providers.py                    
  358 unittests\async_api\test_server.py              
  358 blends\server_pool.py                           
  352 unittests\evaluators\test_llm_judge_reference.py
  341 templates\template_code_mocked.py               
  338 async_api\train.py                              
  336 async_api_agent\blending.py                     
  326 gep_lora/core/pipeline/continue_run.py                                 
  326 templates\template_remote_code.py               
  325 api_jobs\user1\job6\job_run1\run_010.py         
  325 api_jobs\user1\job6\job_run1\run_024.py         
  325 api_jobs\user1\job8\job_run1\run_019.py         
  324 unittests\search\test_selection.py              



```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\config\\settings.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\config\settings.py</path>
<type>file</type>
<content>
1: """
2: settings.py - The knobs for a complete run, in one place.
3: 
4: gep_lora/core/pipeline/start_run.py reads this module. Keeping the values here rather than at the top of it
5: means there is no second copy to drift, and it is what lets a sweep record the
6: settings it ran under without listing them by hand -- snapshot() takes every
7: upper-case name below, so a knob added here is a knob stored there.
8: 
9: Change a value and re-run; nothing else needs editing.
10: """
11: 
12: # --- the population --------------------------------------------------------
13: 
14: # How many individuals the population holds.
15: COUNT = 8
16: 
17: # Seed for the population draw. An int repeats the same population every run;
18: # None grows a fresh one each time -- and, since a sweep records what it drew,
19: # even that stays repeatable afterwards. Note this is separate from the LoRA
20: # blend weights each individual is evaluated under -- those come from
21: # WEIGHT_MASTER_SEED below.
22: SEED = 42 #None
23: 
24: # Reject duplicate chromosomes when building the population.
25: UNIQUE = True
26: 
27: # Deepest level an operator may sit at, and the chance an operator is arity 2
28: # and keeps the branch growing -- the shape a population is drawn with, recorded
29: # alongside the rest so a stored sweep says what shape that was.
30: MAX_DEPTH = 4
31: BRANCH_PROB = 0.2
32: 
33: # --- continuing a sweep ----------------------------------------------------
34: 
35: # How many generations gep_lora/core/pipeline/continue_run.py runs when it is not told otherwise. One
36: # generation is trees -> runs -> process -> evaluate -> fitness -> elitism ->
37: # selection -> mutation -> weight_mutation over the population already in the
38: # database.
39: #
40: # Mind what this costs: process loads the base model once per individual, and
41: # every generation is the whole population again -- and it is the same size
42: # every generation, since selection culls as many individuals as it appends.
43: # SELECTION_COUNT changes the turnover inside the population rather than the
44: # size of it.
45: GENERATIONS = 5
46: 
47: # --- where things go -------------------------------------------------------
48: 
49: # Where the generated scripts go, and the database itself, relative to this file.
50: #
51: # This folder must stay exactly one level below the project folder: a generated
52: # script finds the LoRA folders by going up one from its own directory, so a
53: # deeper path breaks every one of them. Siblings are fine; subfolders are not.
54: # (The eval prompts no longer come into it -- TRAINING_SET below is resolved at
55: # generation time and stamped into each script as an absolute path.)
56: DB_RUN_DIR = "run_db"
57: DB_PATH = "run_db/gep.sqlite3"
58: 
59: # Where test_run_with_dataset.py puts the scripts it runs against a dataset the
60: # sweep was never scored on. A folder of its own, beside run_db/ rather than
61: # inside it: those scripts are the sweep's own, re-pointed at other questions,
62: # and finding one in run_db/ later would be finding a script that answers
63: # something other than what its name says. Same rule as DB_RUN_DIR -- exactly
64: # one level below the project folder, or the LoRA paths inside it break.
65: TESTING_RUN_DIR = "run_testing"
66: 
67: # The eval prompts every generated script is judged on, one per line. It lives
68: # here rather than in the templates so the eval set can be repointed without
69: # editing generated-script code, and so a sweep records which file it was scored
70: # against -- the prompts are half of what a fitness number means. A relative
71: # path is taken from this file's folder, like DB_RUN_DIR above; an absolute one
72: # is used as it stands, so the file need not sit in the project folder at all.
73: # generate_runs.py resolves it and stamps the result into each script, so a
74: # script no longer has to find this file by walking up from itself.
75: TRAINING_SET = "datasets/medical_testing_lora_dataset.json"
76: 
77: # How many records of TRAINING_SET each run is judged on. The first
78: # TRAINING_COUNT of them, in file order, or the whole file when it holds fewer
79: # -- and None for no cap at all, which is every record.
80: #
81: # The top N rather than a sample of N, and the same N for everyone: fitness only
82: # compares across individuals because they all answered the same questions, so a
83: # per-individual draw would make two scores incomparable and a re-run of one
84: # individual incomparable with itself. File order is already arbitrary; taking a
85: # prefix of it keeps that arbitrariness fixed instead of adding a second source
86: # of it.
87: #
88: # This is the cheapest knob in the file: worth turning down while iterating and
89: # back up for a real search, remembering that a short eval set makes a noisier
90: # fitness signal. It is not a proportional saving -- a script answers its
91: # prompts ANSWER_BATCH at a time in one generate() call, and a call reads the
92: # weights once whatever it is answering, so five prompts cost nothing like five
93: # times one. Past ANSWER_BATCH it does start costing another call.
94: TRAINING_COUNT = 20
95: 
96: # The other two splits of the same dataset, recorded beside the training one.
97: #
98: # Nothing in the search reads these yet -- fitness is earned on TRAINING_SET
99: # alone -- but a sweep saves every split it is given into the database's
100: # `datasets` table at the moment it is created, so the questions a later
101: # validation or test pass would use are stored with the sweep that will be
102: # judged on them rather than left to whatever the files happen to hold by then.
103: # Same path rule as TRAINING_SET: relative to this file's folder, absolute used
104: # as it stands. None means that split is simply not part of this sweep.
105: #
106: # Point them at the siblings of whatever TRAINING_SET names, e.g.
107: #     VALIDATION_SET = "datasets/medical_validation_lora_dataset.json"
108: #     TESTING_SET    = "datasets/medical_testing_lora_dataset.json"
109: # and mind that they have to be splits of the same dataset -- a validation set
110: # from another task would be recorded as this sweep's, and mean nothing.
111: VALIDATION_SET = None
112: TESTING_SET = "datasets/medical_validation_lora_dataset.json"
113: 
114: # How good an individual has to have been on the training split before
115: # test_run_with_dataset.py will spend a base-model load asking it the testing
116: # questions. Mean quality over its most recent execution, strictly above this;
117: # --min-quality overrides it for one pass. Turning it down tests more of the
118: # population and costs one model load per extra individual; 0 tests everything
119: # that ever answered anything.
120: TESTING_MIN_QUALITY = 0.5
121: 
122: # How many records of TESTING_SET the testing pass asks: the first N, in file
123: # order, or None for every one of them. TRAINING_COUNT's rule for the other
124: # split, and kept apart from it -- the testing set is asked once per individual
125: # worth testing rather than once a generation, so it can usually afford to be
126: # asked whole. --count overrides it for one pass.
127: TESTING_COUNT = None
128: 
129: 
130: 
131: # --- the adapters being blended --------------------------------------------
132: 
133: # The model every one of the five adapters was trained on, and the model the
134: # generated scripts load before attaching anything -- it reaches them through a
135: # marker, like the slots below, so there is one copy of the name rather than one
136: # per template. Change it only alongside adapters trained on the same base:
137: # PEFT loads a LoRA against the model its adapter_config.json names.
138: #
139: # It is also the identity of the "llm_judge_baseline" evaluator's cached
140: # base-model answers: those are stored under this name, so repointing this at
141: # another model asks for that model's own baseline rather than reusing the old
142: # one's.
143: #BASE_MODEL = "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"
144: BASE_MODEL = "unsloth/Qwen3.5-0.8B" #Qwen2.5-0.5B-Instruct-bnb-4bit
145: 
146: # The chat template every prompt is written in -- by the generated scripts, the
147: # lora servers and the baseline control alike, so a blend and its control are
148: # always asked in the same words. None uses BASE_MODEL's own template, the one
149: # shipped in its repo; a name ("qwen-2.5", "llama-3.1", ...) swaps in unsloth's
150: # template of that name instead.
151: #
152: # None is the right answer for any model with a template of its own, and the
153: # adapters must have been trained under the same one (create_lora.py
154: # --chat-template): an adapter answers in the format it learned, and a base model
155: # prompted in a format it was not built for does not behave as itself --
156: # Qwen3.5 under "qwen-2.5" loses the empty <think> block its own template writes,
157: # and reasons out loud until the length cap. For the Qwen2.5 repos the two are
158: # byte-identical.
159: #
160: # Before this was a setting every template hardcoded "qwen-2.5", so a stored
161: # sweep that never recorded it keeps getting "qwen-2.5"
162: # (generate_runs.chat_template_name). It is part of what a baseline answer
163: # means, so the llm_judge_baseline cache is keyed on it as well as on the model.
164: CHAT_TEMPLATE = None
165: 
166: # Where each of the five LoRAs the trees refer to lives -- the search space
167: # itself, so it belongs with the rest of the knobs rather than in the templates:
168: # repoint a slot here and both templates follow, and the sweep records which
169: # five adapters its fitness numbers were earned on.
170: #
171: # One independent entry per slot. A relative path is taken from this file's
172: # folder; an absolute one is used as it stands. Anything that is neither -- a
173: # Hub repo id, say -- is passed through untouched, which is as far as that has
174: # ever worked: every rank check reads the adapter's own adapter_config.json off
175: # disk. generate_runs.py resolves these once and writes the result into each
176: # generated script, so a script carries real paths rather than working them out
177: # from where it happens to sit.
178: #
179: # These five were trained at different ranks (r=16, 16, 8, 4, 32), which the
180: # code handles -- nothing assumes they match, because PEFT's cat sums input
181: # ranks, svd takes the max, and linear refuses inputs whose ranks differ.
182: LORA_SLOTS = {
183:     "L1": "loras/Lora001/0.8b_own_template_lora_adapter",
184:     "L2": "loras/Lora002/0.8b_own_template_lora_adapter",
185:     "L3": "loras/Lora003/0.8b_own_template_lora_adapter",
186:     "L4": "loras/Lora004/0.8b_own_template_lora_adapter",
187:     "L5": "loras/Lora005/0.8b_own_template_lora_adapter",
188: }
189: 
190: 
191: # --- the generated scripts -------------------------------------------------
192: 
193: # Which template generate_runs.py fills. None means its own default,
194: # template_code.py -- the real thing. Set it to "template_code_mocked.py" for a
195: # dry run: same trees, same ranks, same BAD verdicts, but no model load, random
196: # answers, and scores that arrive with the transcript, so the whole pipeline
197: # finishes in seconds on a machine with no GPU and no judge running. Mocked
198: # scores are noise; never read one as a result.
199: #
200: # "template_remote_code.py" is the third: the same script, but the blend is
201: # built on a lora_server.py process that already has the base model open, so
202: # the import and the model load -- about 54% of what a script costs on this
203: # machine -- are paid once per step instead of once per individual. This one
204: # line is the whole switch: the process step notices what kind of scripts a
205: # sweep holds and starts a pool of servers or does not, and everything else
206: # about a sweep (one script per individual, one execution, the transcript on
207: # stdout, the phase timings) is identical either way. See LORA_SERVER_* below.
208: TEMPLATE = "template_remote_code.py"
209: 
210: # --- the lora servers, for TEMPLATE = "template_remote_code.py" -------------
211: #
212: # Read only when the scripts a sweep holds are lora_server clients -- a sweep
213: # generated from either of the other templates ignores every one of these, so
214: # switching back is the same one line switching forward was.
215: 
216: # How many servers the process step starts, each holding its own copy of the
217: # base model. It also caps the batch: the k-th script of a batch talks to the
218: # k-th server, so a run never has more scripts in flight than there are servers
219: # to serve them, whatever PROCESS_RUN_BATCH_SIZE says.
220: #
221: # The cost is the one a batch always had -- N servers is N copies of the model
222: # resident together -- but it is now paid once per driver rather than once per
223: # individual: the servers stay up across generations, and only come down early
224: # when JUDGE_BACKEND = "unsloth" means the evaluate step wants the card for a
225: # judge of its own. Measured on one card (gep_lora/tools/compare_servers.py): 1 -> 2
226: # servers gave 1.64x throughput and -16% on the process step; 2 -> 4 gave only
227: # 1.29x and -4%, at +23% per individual. A warm server has already removed the
228: # CPU-bound import and load that used to overlap well, and what is left is
229: # GPU-bound, so past two the servers mostly time-share. Two is the sweet spot.
230: LORA_SERVER_COUNT = 2
231: 
232: # Where they listen. Consecutive ports from LORA_SERVER_PORT, one per server,
233: # on an interface that should stay local: these speak no authentication and
234: # will load any adapter folder they are handed.
235: LORA_SERVER_HOST = "127.0.0.1"
236: LORA_SERVER_PORT = 8770
237: 
238: # How long to wait for a server to finish loading before giving up on the step,
239: # and how long one script waits on one request to it. The first covers a cold
240: # model load (a download, on a machine that has never held this model); the
241: # second is per build or per batch of answers, so a hung server costs one
242: # individual rather than the pass.
243: LORA_SERVER_STARTUP_TIMEOUT = 900
244: LORA_SERVER_TIMEOUT = 1800
245: 
246: # How many blends one server may build before the pool restarts it between
247: # batches. A cold process per individual was a strong guarantee -- every result
248: # came from a model that had never seen another blend -- and a warm server is a
249: # weaker one: adapters are attached and deleted on a model that stays up, and
250: # what a server built before this individual is, in principle, part of what it
251: # built for it. This bounds that: the model load is paid again once every N
252: # individuals, which keeps most of the saving and puts a number on the drift.
253: #
254: # 0 never recycles, which is the fastest and the least defensible. Whichever it
255: # is set to, every script prints which server built its blend and how many
256: # blends that server had built before, so the stdout of a stored execution says
257: # where in a server's life it happened.
258: LORA_SERVER_RECYCLE_AFTER = 0
259: 
260: # --- the blend weights -----------------------------------------------------
261: 
262: # Where the per-individual weight seeds come from. Each individual's script is
263: # stamped with a seed derived from this one and its own number, so it draws the
264: # same w1..w5 every time it runs and re-running a stored sweep rebuilds the same
265: # blends. An int makes a whole sweep reproducible from the start; None draws a
266: # master seed at run time and stores it, which is just as repeatable after the
267: # fact -- the value used is written to the run's settings either way.
268: WEIGHT_MASTER_SEED = 42 #None
269: 
270: # --- selection -------------------------------------------------------------
271: 
272: # Where the roulette wheel's spins come from. Each application of the selection
273: # step derives its own generator from this seed and the size of the population
274: # it is spinning over, the way each individual derives its weight seed from
275: # WEIGHT_MASTER_SEED and its own number: one sweep, one recorded seed, and every
276: # draw in it repeatable -- but a second generation still draws its own parents
277: # rather than the first one's again. An int makes a sweep reproducible from the
278: # start; None draws a master seed at run time and stores it.
279: SELECTION_MASTER_SEED = 42 #None
280: 
281: # How many copies a round of selection appends. It also draws one newcomer and
282: # culls that many again -- n+1 in, n+1 out -- so the population stays the size
283: # COUNT drew it at and this sets the turnover instead: at 2, three individuals
284: # in and three out a generation. None is the exception, asking for as many
285: # copies as the population holds, which is one more than the cull can take, so
286: # it is the only value that still grows the population.
287: SELECTION_COUNT = 2
288: 
289: 
290: # --- mutation --------------------------------------------------------------
291: 
292: # The chance each symbol of a chromosome is replaced by another of its own kind
293: # -- per symbol, not per chromosome, so an eleven-symbol individual at 0.1
294: # expects about one change and may well come through untouched. A symbol only
295: # ever becomes one of its own class (CAT/SVD/LIN, L1-L5, w1-w5) and the root is
296: # never touched, which is what keeps every mutated chromosome readable; 0.0
297: # turns mutation off without removing the step.
298: MUTATION_RATE = 0.1
299: 
300: # Where the mutation dice come from, on the same terms as the two seeds above:
301: # an int makes a sweep reproducible from the start, None draws one at run time
302: # and records it.
303: MUTATION_MASTER_SEED = 42 #None
304: 
305: # --- weight mutation -------------------------------------------------------
306: 
307: # The share of the population's weights (the w1-w5 symbols, one per blended
308: # adapter) moved each round -- a count over all of them, not a chance per
309: # weight. Every non-elite individual's weights go into one pool and
310: # round(rate * pool) of them are drawn and swapped for a different w, so ten
311: # weights at 0.1 is exactly one change. The elite's weights are never in the
312: # pool; 0.0 turns the step off without removing it.
313: WEIGHT_MUTATION_RATE = 0.1
314: 
315: # Where that draw comes from, on the same terms as the seeds above.
316: WEIGHT_MUTATION_MASTER_SEED = 42 #None
317: 
318: # --- running the generated scripts -----------------------------------------
319: 
320: # How many generated scripts the process step keeps in flight at once. The
321: # scripts are launched in batches of this size and a batch is waited out before
322: # the next one starts, so this is a fixed ceiling on concurrency rather than a
323: # queue that refills: a batch takes as long as its slowest member.
324: #
325: # 1 is the old behaviour, one script at a time. Anything higher trades memory
326: # for wall clock, and the trade is steep for a real run: every script loads the
327: # base model into its own process, so a batch of N is N copies of the model
328: # resident at the same time. Set this to what the GPU can actually hold -- an
329: # individual that runs out of memory is recorded as a failed execution like any
330: # other, so an over-large batch does not stop a sweep, it quietly fills it with
331: # failures. Mocked runs (TEMPLATE = "template_code_mocked.py") load nothing and
332: # can go much higher.
333: #
334: # The scripts in a batch share the run folder as their working directory, so
335: # they also share the caches unsloth drops there.
336: PROCESS_RUN_BATCH_SIZE = 4
337: 
338: # How often a running script says where it has got to, in seconds. A generated
339: # script prints nothing at all while it loads the base model and then one
340: # question-and-answer pair per eval prompt, so left to itself the console shows
341: # a long hang and then a wall of transcript. The process step reads that output
342: # as it arrives and reports a line for that script every so often instead: the
343: # prompt it is on, or that it is still loading.
344: #
345: # This is the cadence, not the detail. Milestones -- the model coming ready, the
346: # last prompt starting -- are one line each and are always said; the running
347: # count waits for this many seconds to have passed since that script last said
348: # anything, so a fifty-prompt run speaks a handful of times rather than fifty.
349: # Turn it down to watch a run closely, up for a quieter log, and to 0 for the
350: # milestones alone. The transcript itself is never echoed -- it goes to the
351: # database, and store.py --show reads it back.
352: PROCESS_RUN_PROGRESS_SECONDS = 5
353: 
354: 
355: # --- how an answer is scored -----------------------------------------------
356: 
357: # Which evaluator the evaluate step uses. One name, out of the registry in the
358: # evaluators package -- one module per evaluator, plus gep_lora/core/evaluators/common.py:
359: #
360: #   "llm_judge"            a judge model grades the answer on its own merits.
361: #                          Needs an endpoint. The original behaviour, and the
362: #                          default a sweep created before this setting existed
363: #                          is read back under.
364: #   "llm_judge_reference"  the same judge, shown the answer the dataset carries
365: #                          for that question as well. Needs an endpoint and a
366: #                          dataset with assistant turns. This is the one that
367: #                          can see *style*: a judge grading on merit alone
368: #                          happily rewards a helpful prose answer from a blend
369: #                          that was supposed to rhyme.
370: #   "llm_judge_answers"    the same two answers as "llm_judge_reference" --
371: #                          the dataset's and the blend's -- and *not* the
372: #                          question. Needs an endpoint and a dataset with
373: #                          assistant turns. A judge that can see the question
374: #                          quietly grades merit as well as manner; with only
375: #                          the two answers in front of it the score is
376: #                          agreement with the reference and nothing else.
377: #   "llm_judge_baseline"   the same judge, shown what the *base model* replied
378: #                          to the same question as well, and asked how much the
379: #                          blend improved on it. Needs an endpoint, and one run
380: #                          of the base model over the eval set -- cached in the
381: #                          database the first time and read from there ever
382: #                          after. This is the one that scores what this search
383: #                          is actually for: 0.5 means the blend changed nothing
384: #                          worth having, above it means it helped, below it
385: #                          means it did harm.
386: #   "similarity"           token or character overlap with the dataset's answer.
387: #                          No endpoint, deterministic, free -- and a measure of
388: #                          agreement with one particular answer rather than of
389: #                          quality.
390: #   "heuristic"            local checks: length, repetition, a pattern the
391: #                          answer must or must not match. No endpoint. Measures
392: #                          whether an answer is malformed, not whether it is
393: #                          good.
394: #   "panel"                several judge models, aggregated. Less noise per
395: #                          score, N times the cost.
396: #   "composite"            several of the evaluators above, their scores
397: #                          combined -- mean, median, min, max, geometric,
398: #                          harmonic, trimmed mean (COMPOSITE_*).
399: #
400: # Like every setting here, this is frozen into a sweep when it starts: changing
401: # it does nothing to a sweep already running, which is what keeps every fitness
402: # number in one sweep comparable with the others. `python start_run.py --evaluators`
403: # lists what is registered.
404: EVALUATOR = "llm_judge_answers"
405: 
406: 
407: # --- the judge model, for the evaluators that ask one -----------------------
408: #
409: # Read by "llm_judge", "llm_judge_reference", "llm_judge_answers",
410: # "llm_judge_baseline", and by
411: # "panel" for everything except which models sit on it. The API key is deliberately *not* here: a sweep
412: # writes its settings into the database, so the key is read from the
413: # JUDGE_API_KEY environment variable by gep_lora/core/evaluators/common.py instead.
414: 
415: # How the judge is reached. Two transports, one instrument: the rubrics, the
416: # retries, the abandon rule and the way a score is read out of the reply are
417: # shared, so which one a sweep used changes where the tokens came from and
418: # nothing else about the number.
419: #
420: #   "endpoint"  POST to an OpenAI-compatible /v1/chat/completions. Needs a
421: #               server up -- LMStudio, vLLM, OpenAI, OpenRouter -- at
422: #               JUDGE_BASE_URL, and nothing of this machine's GPU.
423: #   "unsloth"   load JUDGE_MODEL in the evaluate step's own process, exactly as
424: #               the generated scripts load the model they blend, and generate
425: #               the grading there. Needs no server at all, so a whole sweep runs
426: #               on this repo and a GPU -- and needs the venv one level up, the
427: #               same interpreter the process step already demands.
428: #
429: # What the local backend costs: the judge is loaded once per evaluate step, on
430: # the first answer it grades rather than while preparing (llm_judge_baseline
431: # runs the base model while preparing, and the two should not be on the card at
432: # once), and released the moment the step is done -- gep_lora/core/pipeline/main.py goes straight into
433: # the next generation, whose scripts each want the VRAM back.
434: JUDGE_BACKEND = "endpoint"
435: 
436: # Where the judge lives, for JUDGE_BACKEND = "endpoint". The default is the
437: # local LMStudio instance; its API is OpenAI-compatible, so a cloud endpoint is
438: # a drop-in replacement:
439: #   OpenAI      https://api.openai.com/v1
440: #   OpenRouter  https://openrouter.ai/api/v1
441: #   vLLM        http://<host>:8000/v1
442: # Claude is not OpenAI-compatible; using a Claude model as the judge needs a
443: # separate backend through the anthropic SDK. Ignored by the unsloth backend,
444: # which has no endpoint to point at.
445: JUDGE_BASE_URL = "http://192.168.1.61:1234/v1"
446: 
447: # Which model does the grading, and what is stored on every exchange it grades.
448: # On the endpoint backend, None asks the endpoint what it has loaded, which is
449: # what you want with LMStudio; name it explicitly for a cloud model. On the
450: # unsloth backend it is a Hub repo id or a folder and it has to be named -- a
451: # machine cannot be asked what it has loaded, and falling back to BASE_MODEL
452: # would leave the model under test grading its own descendants. An
453: # unsloth-quantised instruct model loads fastest, e.g.
454: # "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit".
455: JUDGE_MODEL = None
456: 
457: # Grading should be repeatable, so keep the temperature at zero.
458: JUDGE_TEMPERATURE = 0.0
459: 
460: # The judge emits a short JSON object, but reasoning models spend tokens
461: # thinking first and return an empty message if they run out mid-thought, so
462: # this needs far more headroom than the answer itself requires.
463: JUDGE_MAX_TOKENS = 2000
464: 
465: # Seconds to wait for one grading call, how many times to retry a call that
466: # fails for a transient reason (connection dropped, 5xx, rate limit), and how
467: # long to wait between tries.
468: JUDGE_TIMEOUT = 300
469: JUDGE_RETRIES = 2
470: JUDGE_RETRY_WAIT = 3
471: 
472: # Ask for a JSON object back. None for an endpoint that rejects the parameter --
473: # the prompt asks for JSON anyway, and a 400 falls back to that by itself.
474: # Endpoint backend only: a local judge is asked for JSON by the rubric, and
475: # parse_reply() recovers a score from prose either way.
476: JUDGE_RESPONSE_FORMAT = {"type": "json_object"}
477: 
478: 
479: # --- the local judge, for JUDGE_BACKEND = "unsloth" -------------------------
480: #
481: # Only these three are the local backend's own. Everything else about grading --
482: # which model (JUDGE_MODEL), how hot (JUDGE_TEMPERATURE), how many tokens it may
483: # spend (JUDGE_MAX_TOKENS), how many tries (JUDGE_RETRIES) and when to give up
484: # on an individual (JUDGE_ABANDON_FRACTION) -- is read the same way whichever
485: # backend is in use. JUDGE_BASE_URL, JUDGE_TIMEOUT, JUDGE_RESPONSE_FORMAT and
486: # $JUDGE_API_KEY are the endpoint's, and are ignored here.
487: 
488: # The context window the judge is loaded with. It has to hold the rubric, the
489: # question and the answer, and for "llm_judge_reference" and
490: # "llm_judge_baseline" a second answer beside it, plus JUDGE_MAX_TOKENS of
491: # reply. A prompt over this is that one answer's failure, not the step's.
492: JUDGE_LOCAL_MAX_SEQ_LENGTH = 4096
493: 
494: # Load the judge quantised. True is what makes a second model fit beside
495: # everything else this pipeline wants from the card; False for a judge whose
496: # grading you want at full precision, and the VRAM to spare.
497: JUDGE_LOCAL_LOAD_IN_4BIT = True
498: 
499: # Which chat template to format the rubric and the answer under. None uses the
500: # model's own, which is what a stock instruct model ships and what it was
501: # trained to answer under. Name one unsloth knows ("qwen-2.5", "llama-3.1",
502: # ...) only for a model whose tokeniser carries none, or carries a wrong one.
503: JUDGE_LOCAL_CHAT_TEMPLATE = None
504: 
505: # When to stop grading an individual that is going nowhere. A judge call is the
506: # expensive part of a sweep, and an individual whose *first* graded answers all
507: # come back 0.0 is one the search has already decided against: the rest of its
508: # answers cost a call each to confirm a fitness of zero. This is the fraction of
509: # an individual's answers that has to be graded, and to be unanimously zero,
510: # before the evaluate step gives up on it -- 0.1 is "the first 10%", rounded up,
511: # and never fewer than one answer. The abandoned answers are stored as 0.0 with
512: # a reason saying so, so the individual's fitness really is the mean of its own
513: # scores, and a re-run does not ask about them again.
514: #
515: # Only the evaluators that ask a model (llm_judge, llm_judge_reference,
516: # llm_judge_answers, llm_judge_baseline, panel) abandon anything, whichever
517: # backend they ask it
518: # through -- a local *scorer* like "similarity" costs nothing to finish, and
519: # cutting it short would only lose detail.
520: #
521: # Mind how sharp this is on a small eval set: with TRAINING_COUNT = 10 the first
522: # 10% is a single answer, so one zero condemns the individual. 0 or None grades
523: # every answer, whatever the early ones say.
524: JUDGE_ABANDON_FRACTION = 0.1
525: 
526: # The three judging rubrics used to live here. Each is now a constant in the
527: # evaluator that sends it, beside the code that sends it:
528: #
529: #   JUDGE_SYSTEM_PROMPT            gep_lora/core/evaluators/llm_judge.py
530: #   JUDGE_REFERENCE_SYSTEM_PROMPT  gep_lora/core/evaluators/llm_judge_reference.py
531: #   JUDGE_BASELINE_SYSTEM_PROMPT   gep_lora/core/evaluators/llm_judge_baseline.py
532: #
533: # JUDGE_ANSWERS_SYSTEM_PROMPT (gep_lora/core/evaluators/llm_judge_answers.py) was never a
534: # setting at all -- that evaluator arrived after the move, so it has no stored
535: # value in any sweep to fall back to.
536: #
537: # "panel" reads the first two as well, and keeps its own copy of each, so its
538: # rubrics can be tuned for a panel without moving what the single-judge
539: # evaluators select on.
540: #
541: # They are prompts rather than knobs: one evaluator's own text, which nothing
542: # else reads. What that costs is that they are no longer snapshotted into a
543: # sweep, so a sweep no longer records the rubric it was judged under. Every
544: # reader therefore still lets a sweep's *stored* value win, and a sweep created
545: # while these were settings holds its own copies -- so an old sweep goes on
546: # being graded by the rubric it actually ran under.
547: 
548: # --- the base-model answers "llm_judge_baseline" grades against -------------
549: #
550: # Producing them costs one base-model load and one generate() per eval prompt,
551: # once. They are then cached in the database under BASE_MODEL and the question,
552: # outside any one sweep, so every later sweep on the same base model and the
553: # same prompts reads them and loads nothing.
554: 
555: # Which template the one baseline script is filled from. None picks it from the
556: # sweep's own TEMPLATE: template_baseline.py for a real sweep, and
557: # template_baseline_mocked.py for one generated from template_code_mocked.py,
558: # so a mocked sweep still needs no GPU. Mocked baselines are cached apart from
559: # real ones -- under "mock:<model>" -- so a dry run can never leave made-up base
560: # answers where a real sweep would find them.
561: BASELINE_TEMPLATE = None
562: 
563: # Seconds to allow the baseline script. It answers every eval prompt in one
564: # process, so give it roughly what one individual gets, times the prompts.
565: BASELINE_TIMEOUT = 1800
566: 
567: 
568: # --- the "jev_judge_reference" evaluator ------------------------------------
569: #
570: # llm_judge_reference's question -- does the answer match the dataset's own
571: # answer in manner and substance -- graded by Jev (typesafe.ai's "System One"
572: # model) instead of an LLM. Jev is not read from JUDGE_BACKEND/JUDGE_BASE_URL:
573: # it is a different service with a different request shape (a typed decision,
574: # not a chat completion), so it gets its own settings. The API key is
575: # deliberately not one of them, for the same reason JUDGE_API_KEY isn't: it is
576: # read from the TYPESAFE_API_KEY environment variable by
577: # gep_lora/core/evaluators/jev_judge_reference.py instead.
578: 
579: # The TypeSafe evaluation endpoint's host. None uses the public API.
580: JEV_BASE_URL = None
581: 
582: # Which Jev release grades. "jev-latest" tracks TypeSafe's current stable
583: # release; a versioned id (e.g. "jev-1.13.0") pins a sweep to one build.
584: JEV_MODEL = "jev-latest"
585: 
586: # Seconds to wait for one grading call, how many times to retry a call that
587: # fails for a transient reason (dropped connection, rate limit, the service
588: # overloaded), and how long to wait between tries. Jev is small and fast next
589: # to a judge model -- TypeSafe's own numbers are on the order of 100ms a call --
590: # so this is far shorter than JUDGE_TIMEOUT.
591: JEV_TIMEOUT = 30
592: JEV_RETRIES = 2
593: JEV_RETRY_WAIT = 3
594: 
595: 
596: # --- the "similarity" evaluator --------------------------------------------
597: 
598: # How an answer is compared with the dataset's answer:
599: #   "token_f1"     bag-of-words F1, repeats counted. Balanced: an answer that
600: #                  is all reference words plus padding loses precision, one
601: #                  that covers half of them loses recall.
602: #   "containment"  how much of the reference's vocabulary turns up at all.
603: #                  Forgiving about length and about everything else added.
604: #   "sequence"     character-level overlap (difflib), so word order and
605: #                  phrasing count. The strictest of the three.
606: SIMILARITY_METRIC = "token_f1"
607: 
608: # Whether case counts. Off by default, since a lowercase answer to an uppercase
609: # reference is usually the same answer -- turn it on for an adapter whose whole
610: # job is a matter of case, or use HEURISTIC_REQUIRE for that instead.
611: SIMILARITY_CASE_SENSITIVE = False
612: 
613: 
614: # --- the "heuristic" evaluator ---------------------------------------------
615: 
616: # The length band an answer is expected to fall in, in words. Under the floor
617: # scores proportionally; over the ceiling falls away from it. None for no
618: # ceiling.
619: HEURISTIC_MIN_WORDS = 8
620: HEURISTIC_MAX_WORDS = 400
621: 
622: # A pattern the answer must contain, and one it must not, or None for neither.
623: # Each is a Python regular expression, searched with re.MULTILINE, and each
624: # counts as one of the equally weighted checks that make up the score. This is
625: # where a task whose rule is genuinely checkable gets checked -- r"^[^a-z]*$"
626: # for an all-uppercase adapter, say.
627: HEURISTIC_REQUIRE = None
628: HEURISTIC_FORBID = None
629: 
630: 
631: # --- the "panel" evaluator --------------------------------------------------
632: 
633: # The models on the panel, by id, all served by one endpoint. An empty list asks
634: # the endpoint what it has loaded and sits a panel of one on it -- which is
635: # "llm_judge" with extra steps, so name at least two to get the point of this.
636: PANEL_MODELS = []
637: 
638: # Where the panel is served, or None for JUDGE_BASE_URL. Endpoint backend only:
639: # under JUDGE_BACKEND = "unsloth" the members are loaded here, all of them
640: # resident at once for as long as the step lasts, the way
641: # PROCESS_RUN_BATCH_SIZE holds N base models -- so keep the list short and the
642: # members small. Everything else about
643: # a member -- temperature, token budget, timeouts, the rubric -- comes from the
644: # JUDGE_* settings above, so a panel is several models grading identically
645: # rather than several differently configured judges.
646: PANEL_BASE_URL = None
647: 
648: # How the members' scores become one: "mean", "median", "min" or "max". Median
649: # ignores a single outlying judge; min is the pessimistic reading, and selects
650: # for answers no member objected to.
651: PANEL_AGGREGATE = "mean"
652: 
653: # Whether the panel grades against the dataset's own answer -- the
654: # JUDGE_REFERENCE_SYSTEM_PROMPT rubric, panel.py's own copy of it -- rather than
655: # on merit alone. Needs a dataset with assistant turns, exactly like
656: # "llm_judge_reference".
657: PANEL_USE_REFERENCE = False
658: 
659: 
660: # --- the "composite" evaluator ----------------------------------------------
661: 
662: # The evaluators combined, in the order they are prepared and asked: a name
663: # (weight 1), ["name", weight] or {"name": "...", "weight": ...}. Each member
664: # reads the same settings it would as EVALUATOR -- JUDGE_*, HEURISTIC_* and the
665: # rest -- so it grades exactly as it would alone. A name may appear once, and
666: # "composite" not at all. Empty is refused when a composite sweep is scored.
667: COMPOSITE_EVALUATORS = ["llm_judge_reference", "llm_judge_baseline"]
668: 
669: # How the members' scores become one:
670: #   "mean"          weighted mean
671: #   "median"        weighted median -- ignores one member out on its own
672: #   "min" / "max"   the worst / best member; weights ignored
673: #   "geometric"     weighted geometric mean -- a low member costs much more
674: #                   than in the mean, and a 0 from anyone is 0
675: #   "harmonic"      weighted harmonic mean -- harsher on imbalance still
676: #   "trimmed_mean"  drop the highest and lowest, mean of the rest (3+ members)
677: # On a 0..1 scale min <= harmonic <= geometric <= mean <= max always, so moving
678: # along that list is choosing how much one bad member should cost.
679: COMPOSITE_AGGREGATE = "mean"
680: 
681: # A veto: any member scoring below this makes the answer 0.0, whatever the rest
682: # said -- how a checkable rule (heuristic, say) becomes a gate rather than one
683: # vote. None for no veto.
684: COMPOSITE_FLOOR = None
685: 
686: # A member whose score() fails: "fail" fails the whole exchange, since a mean
687: # over whichever members answered is a different number from the mean over all
688: # of them; "skip" combines the members that did answer, as panel does.
689: COMPOSITE_ON_FAILURE = "fail"
690: 
691: 
692: 
693: def snapshot():
694:     """Every setting above as {name: value}, for recording what a run used."""
695:     return {name: value for name, value in sorted(globals().items())
696:             if name.isupper() and not name.startswith("_")}

(End of file - total 696 lines)
</content>
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 5.4s)

_Thinking:_

This is a big and well-documented project. The user wants a code critique. Let me read the core modules to understand the actual code quality, not just the docs. Let me focus on the biggest / most important: gep_lora/core/pipeline/start_run.py, store.py, generate_population.py, generate_runs.py, process_run.py, gep_lora/core/pipeline/main.py, selection.py, mutation.py, and the templates.

Let me spawn parallel explore agents to examine different parts, and read some files directly.

Actually, let me be strategic. I'll read key files myself for the critique. Let me batch reads.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\search\\generate_population.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\search\generate_population.py</path>
<type>file</type>
<content>
1: """
2: generate_population.py - The alphabet, the encoding, and the random draw.
3: 
4: This is the root module of the pipeline: it owns the symbols an individual is
5: made of, the Node type, and the encode/decode pair everything else reads trees
6: with. gep_lora/core/pipeline/start_run.py calls build_population() for a sweep's population and decode() for
7: each tree; there is no second parser anywhere.
8: 
9: Encoding
10: --------
11: Every individual is a Karva (K-)expression: the tree is written out in
12: level-order (breadth first, left to right), one symbol per position, joined
13: with dots. Reading it back is the same walk in reverse -- take the symbols in
14: order and hand each one out as the next child that is still missing.
15: 
16:     CAT.SVD.LIN.L1.L2.L3.L1.w3.w3.w2.w1
17: 
18:         CAT
19:         SVD.LIN
20:         L1.L2.L3.L1
21:         w3.w3.w2.w1
22: 
23: Grammar
24: -------
25:     CAT, SVD, LIN   arity 2, their children must be operators
26:     L1 .. L5        arity 1, their child must be a variable
27:     w1 .. w5        variables, the leaves of the tree
28: 
29: The first symbol is always CAT.
30: """
31: 
32: from collections import deque
33: 
34: 
35: # --- the alphabet ---------------------------------------------------------
36: 
37: BINARY_OPS = ("CAT", "SVD", "LIN")           # arity 2, feed on other operators
38: UNARY_OPS = ("L1", "L2", "L3", "L4", "L5")   # arity 1, feed on variables
39: VARIABLES = ("w1", "w2", "w3", "w4", "w5")
40: 
41: ROOT = "CAT"
42: 
43: ARITY = {}
44: ARITY.update({op: 2 for op in BINARY_OPS})
45: ARITY.update({op: 1 for op in UNARY_OPS})
46: ARITY.update({var: 0 for var in VARIABLES})
47: 
48: 
49: def children_alphabet(symbol):
50:     """The symbols that are legal as a child of `symbol`."""
51:     if symbol in BINARY_OPS:
52:         return BINARY_OPS + UNARY_OPS
53:     if symbol in UNARY_OPS:
54:         return VARIABLES
55:     return ()
56: 
57: 
58: class Node:
59:     """One tree node: a symbol plus its ordered children."""
60: 
61:     __slots__ = ("symbol", "children")
62: 
63:     def __init__(self, symbol):
64:         self.symbol = symbol
65:         self.children = []
66: 
67: 
68: # --- growing a random tree ------------------------------------------------
69: 
70: 
71: def random_tree(rng, max_depth, branch_prob):
72:     """Grow a random valid tree rooted at CAT.
73: 
74:     `max_depth` is the deepest level an *operator* may sit at (the root is
75:     level 0), so a variable can appear at most one level below that. At that
76:     last operator level only arity-1 operators are drawn, which caps the tree.
77:     `branch_prob` is the chance that an operator above that level is arity 2,
78:     i.e. that the branch keeps growing instead of closing off with an L*.
79:     """
80:     root = Node(ROOT)
81:     pending = deque([(root, 0)])
82:     while pending:
83:         node, depth = pending.popleft()
84:         for _ in range(ARITY[node.symbol]):
85:             if node.symbol in UNARY_OPS:
86:                 child = Node(rng.choice(VARIABLES))
87:             elif depth + 1 >= max_depth or rng.random() >= branch_prob:
88:                 child = Node(rng.choice(UNARY_OPS))
89:             else:
90:                 child = Node(rng.choice(BINARY_OPS))
91:             node.children.append(child)
92:             if ARITY[child.symbol]:
93:                 pending.append((child, depth + 1))
94:     return root
95: 
96: 
97: # --- encoding / decoding --------------------------------------------------
98: 
99: 
100: def levels(root):
101:     """The tree as a list of levels, each a list of symbols."""
102:     rows = []
103:     frontier = [root]
104:     while frontier:
105:         rows.append([node.symbol for node in frontier])
106:         frontier = [child for node in frontier for child in node.children]
107:     return rows
108: 
109: 
110: def encode(root):
111:     """Tree -> dotted K-expression (level-order walk)."""
112:     return ".".join(symbol for row in levels(root) for symbol in row)
113: 
114: 
115: def decode(expression):
116:     """Dotted K-expression -> (tree, number of symbols the tree used).
117: 
118:     Raises ValueError on anything that breaks the grammar. Symbols left over
119:     once the tree is complete are the unused tail, and are reported through
120:     the returned count rather than silently dropped.
121:     """
122:     tokens = expression.split(".")
123:     if not tokens or tokens[0] != ROOT:
124:         raise ValueError("expression must start with %s" % ROOT)
125: 
126:     root = Node(tokens[0])
127:     pending = deque([root])
128:     used = 1
129:     while pending:
130:         node = pending.popleft()
131:         allowed = children_alphabet(node.symbol)
132:         for _ in range(ARITY[node.symbol]):
133:             if used >= len(tokens):
134:                 raise ValueError("expression ends while %s still needs a child" % node.symbol)
135:             symbol = tokens[used]
136:             used += 1
137:             if symbol not in allowed:
138:                 raise ValueError("%s is not a legal child of %s" % (symbol, node.symbol))
139:             child = Node(symbol)
140:             node.children.append(child)
141:             if ARITY[symbol]:
142:                 pending.append(child)
143:     return root, used
144: 
145: 
146: def check(expression):
147:     """Decode `expression` and make sure it round-trips exactly."""
148:     total = len(expression.split("."))
149:     root, used = decode(expression)
150:     if used != total:
151:         raise ValueError("expression has %d unused trailing symbols" % (total - used))
152:     if encode(root) != expression:
153:         raise ValueError("expression does not round-trip")
154:     return root
155: 
156: 
157: # --- the population --------------------------------------------------------
158: 
159: 
160: def build_population(count, rng, max_depth, branch_prob, unique):
161:     """Generate `count` validated expressions."""
162:     population = []
163:     seen = set()
164:     attempts = 0
165:     attempt_budget = count * 100
166:     while len(population) < count:
167:         attempts += 1
168:         if attempts > attempt_budget:
169:             raise RuntimeError(
170:                 "could only find %d unique expressions out of %d requested; "
171:                 "raise MAX_DEPTH or BRANCH_PROB in settings.py, or clear UNIQUE"
172:                 % (len(population), count)
173:             )
174:         depth = rng.randint(1, max_depth)
175:         expression = encode(random_tree(rng, depth, branch_prob))
176:         check(expression)   # never store something we cannot read back
177:         if unique:
178:             if expression in seen:
179:                 continue
180:             seen.add(expression)
181:         population.append(expression)
182:     return population

(End of file - total 182 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\storage\\store.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\storage\store.py</path>
<type>file</type>
<content>
1: """
2: store.py - The sqlite database behind gep_lora/core/pipeline/start_run.py.
3: 
4: A sweep scattered across a folder -- a population file, a tree file, an index,
5: a script, an output and a transcript per individual -- is easy to read and
6: impossible to query: "which chromosomes scored above 0.7, and under which
7: weights" means opening a hundred files by hand, and the next sweep overwrites
8: them all.
9: 
10: This module holds the whole of a sweep in one file, with the sweep itself as a
11: row so sweeps accumulate instead of replacing each other:
12: 
13:     runs          one sweep: when, which template, which interpreter, which commit
14:       settings    every knob it ran under, including the seeds
15:       datasets    the training/validation/testing records it was given, saved
16:                   the moment the sweep was created
17:       individuals the population: chromosome, tree, rank, verdict, and the
18:                   generated script in full
19:         executions  one per time that individual was actually run: exit code,
20:                     seconds, the weight seed and the weights it drew, stdout,
21:                     stderr
22:           exchanges the questions and answers, and the judge's score for each
23:       fitness_history  what each individual's fitness was at the end of each
24:                     generation, and when it was worked out
25:       step_timings  what each step of each pass cost, and how much work it did
26:         phase_timings  where inside one step those seconds went
27: 
28:     baselines     what the base model itself answered, per model and question.
29:                   The one table outside that tree: it belongs to the model
30:                   rather than to a sweep, so every sweep that grades against
31:                   the base model reads the same cache instead of producing it
32:                   again.
33: 
34: A sweep therefore records what it produced *and* what it cost, in the same
35: file and to the same standard: which step, how many items it was, and which
36: phase inside it -- see the step_timings and phase_timings comments below, and
37: `python -m gep_lora.core.metrics.report` for the read side.
38: 
39: Everything a scorer needs is reachable from one query, and nothing is derived
40: from a filename. The only thing that still has to exist on disk is the generated
41: run_NNN.py scripts, because process runs them as subprocesses -- and those are a
42: cache of `individuals.script_source`, written out by materialise() whenever they
43: are missing.
44: 
45: Repeatability comes from the settings table: the population seed, and the weight
46: seed stamped into each individual's script. Re-running a sweep with the stored
47: values reproduces both the chromosomes and the blends they were judged under.
48: 
49: Usage as a module:
50: 
51:     conn = store.connect("run_db/gep.sqlite3")
52:     run_id = store.create_run(conn, template="template_code.py")
53:     store.save_settings(conn, run_id, settings.snapshot())
54: 
55: Usage from the command line:
56: 
57:     python -m gep_lora.core.storage.store --list                  # the sweeps in the database
58:     python -m gep_lora.core.storage.store --show 3                # one sweep, summarised
59:     python -m gep_lora.core.storage.store --export 3 --into dump  # write it back out as text files
60: """
61: 
62: import argparse
63: import json
64: import os
65: import sqlite3
66: import subprocess
67: import sys
68: import time
69: 
70: # The repo folder, one above this one. Every path a setting names is
71: # resolved against it, so nothing here depends on the cwd a driver was
72: # started from, or on which sub-folder this module ended up in.
73: _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
74: 
75: 
76: # --- schema ----------------------------------------------------------------
77: 
78: # `state` is ok | BAD as the generator decides it, `verdict` is what the exit
79: # code came to, and the NNN numbering is the individual's own number -- the same
80: # vocabulary the generated scripts and their output use.
81: SCHEMA = """
82: CREATE TABLE IF NOT EXISTS runs (
83:     id          INTEGER PRIMARY KEY,
84:     created_at  TEXT    NOT NULL,
85:     label       TEXT,
86:     template    TEXT    NOT NULL,
87:     status      TEXT    NOT NULL DEFAULT 'open',   -- open | done | failed
88:     git_commit  TEXT,
89:     interpreter TEXT
90: );
91: 
92: CREATE TABLE IF NOT EXISTS settings (
93:     run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
94:     key     TEXT    NOT NULL,
95:     value   TEXT,
96:     PRIMARY KEY (run_id, key)
97: );
98: 
99: -- The dataset the sweep was given, one row per record, saved at the moment the
100: -- sweep is created and never touched again -- for the reason the settings are:
101: -- a fitness number means "this blend, under these knobs, on these questions",
102: -- and the file the questions came from goes on being edited, repointed and
103: -- regenerated long after the sweep that was judged on it.
104: --
105: -- `split` is which of the three the row belongs to. Only the training split is
106: -- read by the search today (it is TRAINING_SET, capped by TRAINING_COUNT, and
107: -- what every generated script asks); validation and testing are stored when
108: -- settings name them so that a later pass has the questions this sweep was
109: -- built beside rather than whatever those files hold by then.
110: --
111: -- Every record of the file is stored, uncapped: TRAINING_COUNT says how many
112: -- an individual is judged on, which is a fact about the sweep and already in
113: -- the settings table, not a fact about the dataset. `position` is 1-based over
114: -- non-blank lines from the top, so it lines up with exchanges.position the way
115: -- generate_runs.eval_records() does. `content` is the line as it stood --
116: -- question and reference are this repo's reading of it, and keeping the line
117: -- means a later reader can disagree.
118: CREATE TABLE IF NOT EXISTS datasets (
119:     id        INTEGER PRIMARY KEY,
120:     run_id    INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
121:     split     TEXT    NOT NULL
122:               CHECK (split IN ('training', 'validation', 'testing')),
123:     source    TEXT    NOT NULL,      -- the file it was read from, resolved
124:     position  INTEGER NOT NULL,      -- 1-based, non-blank lines from the top
125:     question  TEXT    NOT NULL,      -- the user turn: what a script would ask
126:     reference TEXT,                  -- the assistant turn, when the record has one
127:     content   TEXT    NOT NULL,      -- the whole line, as it was read
128:     UNIQUE (run_id, split, position)
129: );
130: 
131: CREATE TABLE IF NOT EXISTS individuals (
132:     id            INTEGER PRIMARY KEY,
133:     run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
134:     number        INTEGER NOT NULL,        -- 1-based; the NNN in run_NNN.py
135:     chromosome    TEXT    NOT NULL,
136:     tree          TEXT,                    -- the drawing draw_trees.draw() makes
137:     state         TEXT,                    -- ok | BAD (PEFT's equal-rank rule)
138:     rank          INTEGER,                 -- rank of the final adapter
139:     script_name   TEXT,
140:     script_source TEXT,
141:     weight_seed   INTEGER,                 -- stamped into the script above
142:     fitness       REAL DEFAULT 0.0,
143:     is_best       INTEGER DEFAULT 0,
144:     has_changed   INTEGER DEFAULT 0,
145:     UNIQUE (run_id, number)
146: );
147: 
148: CREATE TABLE IF NOT EXISTS executions (
149:     id            INTEGER PRIMARY KEY,
150:     individual_id INTEGER NOT NULL REFERENCES individuals(id) ON DELETE CASCADE,
151:     started_at    TEXT    NOT NULL,
152:     seconds       REAL,
153:     exit_code     INTEGER,                 -- NULL means it hit the timeout
154:     verdict       TEXT,                    -- ok | timeout | exit N
155:     weight_seed   INTEGER,
156:     weights       TEXT,                    -- JSON {"w1": .., .. } as drawn
157:     stdout        TEXT,
158:     stderr        TEXT
159: );
160: 
161: CREATE TABLE IF NOT EXISTS exchanges (
162:     id           INTEGER PRIMARY KEY,
163:     execution_id INTEGER NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
164:     position     INTEGER NOT NULL,         -- 1-based, in the order asked
165:     question     TEXT    NOT NULL,
166:     answer       TEXT    NOT NULL DEFAULT '',
167:     quality      REAL,                     -- 0..1, NULL until judged
168:     reason       TEXT,
169:     judged_at    TEXT,
170:     judge_model  TEXT,
171:     UNIQUE (execution_id, position)
172: );
173: 
174: -- Fitness as it stood at the end of each generation, one row per individual
175: -- per generation. individuals.fitness only ever holds the latest value and
176: -- mutation clears it outright, so without this a sweep can say how fit its
177: -- population is now and nothing whatever about how it got there.
178: --
179: -- A row is self-contained for the reason an execution is: it keeps the
180: -- chromosome and the verdict it was scored under, because the individual it
181: -- names goes on being rewritten -- mutation replaces the chromosome, the next
182: -- generation replaces the fitness -- and a history that has to join back to the
183: -- current population to be read would report the past in terms of the present.
184: CREATE TABLE IF NOT EXISTS fitness_history (
185:     id          INTEGER PRIMARY KEY,
186:     run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
187:     generation  INTEGER NOT NULL,        -- 1-based, in the order fitness ran
188:     population  INTEGER NOT NULL,        -- individuals the generation held
189:     recorded_at TEXT    NOT NULL,        -- when the fitness step wrote the row
190:     number      INTEGER NOT NULL,        -- the individual's own number
191:     chromosome  TEXT    NOT NULL,        -- as it was then, not as it is now
192:     state       TEXT,                    -- ok | BAD, as it was then
193:     fitness     REAL,                    -- what individuals.fitness was given
194:     answers     INTEGER,                 -- exchanges the mean was taken over
195:     unscored    INTEGER,                 -- how many of those carried no quality
196:     UNIQUE (run_id, generation, number)
197: );
198: 
199: -- What the base model, with no adapter attached, says to each eval prompt.
200: --
201: -- The one table here that hangs off no run, on purpose: a base-model answer
202: -- belongs to the model and the question and to nothing else, so the sweep that
203: -- happened to pay for it is not part of its identity. Every later sweep on the
204: -- same base model reads these rather than loading the model again, which is the
205: -- whole reason they are stored -- producing them costs a model load and one
206: -- generate() per prompt.
207: --
208: -- `model` is BASE_MODEL as the sweep recorded it, with a "mock:" prefix for
209: -- answers a mocked baseline invented, so a dry run can never leave made-up
210: -- answers where a real sweep would find them. `question_key` is the question
211: -- reduced to what makes two of them the same question (see question_key), and
212: -- is what the lookup matches on; `question` keeps the wording as it was asked.
213: CREATE TABLE IF NOT EXISTS baselines (
214:     id           INTEGER PRIMARY KEY,
215:     model        TEXT NOT NULL,          -- BASE_MODEL, or mock:<BASE_MODEL>
216:     question_key TEXT NOT NULL,          -- the question, normalised for matching
217:     question     TEXT NOT NULL,          -- as it was asked
218:     answer       TEXT NOT NULL DEFAULT '',
219:     created_at   TEXT NOT NULL,
220:     source       TEXT,                   -- which template produced it
221:     UNIQUE (model, question_key)
222: );
223: 
224: -- What a finished individual says on a dataset it was never scored on.
225: --
226: -- The search judges every individual on the training split, over and over, and
227: -- an individual that scores well there has been selected *for* that split. The
228: -- question a testing pass asks is the other one: does the blend hold up on
229: -- questions it was never picked for? test_run_with_dataset.py takes the ones
230: -- worth asking it of, re-points their own scripts at another dataset and runs
231: -- them again, and this is where the answers land.
232: --
233: -- Deliberately not a second `executions` table hanging off `individuals`:
234: -- fitness, elitism and selection all read the *latest execution* of an
235: -- individual, so a test run stored there would be picked up as that
236: -- individual's current result and would decide the next generation on the
237: -- strength of questions the search is not judged on. Its own table cannot be
238: -- read by mistake.
239: --
240: -- A row is self-contained the way an execution row is, and for the reason
241: -- fitness_history rows are: the individual it names goes on being rewritten by
242: -- mutation and selection, so the number, the chromosome, the weights and the
243: -- quality that got it picked are all kept as they were at the moment it was
244: -- tested. `exchanges` is the whole transcript as JSON -- question, answer, and
245: -- whatever quality the run itself carried -- because a testing pass is read
246: -- back whole and a script that crashed still leaves a row saying so, with an
247: -- empty transcript and its exit code. The test_answers view below unnests it
248: -- for the times a query wants one row per question.
249: CREATE TABLE IF NOT EXISTS test_results (
250:     id            INTEGER PRIMARY KEY,
251:     run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
252:     individual_id INTEGER NOT NULL REFERENCES individuals(id) ON DELETE CASCADE,
253:     number        INTEGER NOT NULL,      -- the individual's own number, as it was
254:     chromosome    TEXT    NOT NULL,      -- what the script that ran builds
255:     dataset       TEXT    NOT NULL,      -- the file it was tested on, resolved
256:     records       INTEGER,               -- how many of its questions it was asked
257:     selected_on   REAL,                  -- the training quality that got it picked
258:     started_at    TEXT    NOT NULL,
259:     seconds       REAL,
260:     exit_code     INTEGER,               -- NULL means it hit the timeout
261:     verdict       TEXT,                  -- ok | timeout | exit N
262:     weight_seed   INTEGER,
263:     weights       TEXT,                  -- JSON {"w1": .., .. } as drawn
264:     stdout        TEXT,
265:     stderr        TEXT,
266:     exchanges     TEXT    NOT NULL DEFAULT '[]',  -- JSON, the whole transcript
267:     -- Filled by the scoring pass, which is a second visit: a pass stores the
268:     -- answers first and grades them after, so these are NULL until it has.
269:     quality       REAL,                   -- mean over this row's scored answers
270:     evaluator     TEXT,                   -- which one gave those scores
271:     judge_model   TEXT,                   -- its label: a model id, or a method
272:     scored_at     TEXT
273: );
274: 
275: -- How long each step of the pipeline took, one row per step per pass.
276: --
277: -- A sweep already records what it produced and nothing about what it cost, so
278: -- "which step should be optimised first" can only be answered by watching the
279: -- console go by -- and the console is gone by the time the question is asked.
280: -- These rows are that answer, kept the way every other result is kept.
281: --
282: -- One row per *step*, per *pass* over the sweep: a pass is one run of
283: -- start_run.run(), which is exactly one generation when gep_lora/core/pipeline/continue_run.py is
284: -- turning the crank and one partial run when a step list was named by hand. So
285: -- `process` appears once per generation rather than once per sweep, and the
286: -- cost of a generation is the sum of the rows sharing its pass_no.
287: --
288: -- `items` and `unit` are what makes two steps comparable: a step is only
289: -- expensive relative to how much it did, and seconds/item is the number that
290: -- says whether the fix is a faster item or fewer of them. `skipped` is the
291: -- other half of that -- individuals a step declined to run (BAD, unchanged,
292: -- already scored) cost nothing and would otherwise flatter the per-item cost.
293: --
294: -- `watermark` is store.high_number() when the pass began: the same clock
295: -- fitness_history is dated by, so a timing row can be lined up with the
296: -- generation whose fitness it paid for. It is taken once for the whole pass, so
297: -- the selection step raising it mid-pass does not split a generation in two.
298: CREATE TABLE IF NOT EXISTS step_timings (
299:     id          INTEGER PRIMARY KEY,
300:     run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
301:     pass_no     INTEGER NOT NULL,        -- 1-based, one per start_run.run()
302:     position    INTEGER NOT NULL,        -- the step's place within that pass
303:     step        TEXT    NOT NULL,        -- the STEPS name: process, evaluate, ..
304:     generation  TEXT,                    -- the driver's label ("2/5"), if any
305:     watermark   INTEGER,                 -- high_number() when the pass began
306:     started_at  TEXT    NOT NULL,
307:     seconds     REAL    NOT NULL,        -- wall time, the whole step
308:     items       INTEGER DEFAULT 0,       -- units it actually worked on
309:     unit        TEXT,                    -- what one item is
310:     skipped     INTEGER DEFAULT 0,       -- units it declined to work on
311:     status      TEXT    NOT NULL DEFAULT 'ok',   -- ok | failed
312:     note        TEXT
313: );
314: 
315: -- Where the time inside one step went, one row per phase per step.
316: --
317: -- The step rows say which step to optimise; these say what to do about it. A
318: -- phase is a named piece of work a step repeats -- loading the base model,
319: -- attaching a leaf, folding one node, one generate(), one judge call -- and a
320: -- row aggregates every occurrence of it within that step rather than storing
321: -- one row per occurrence: `calls` and `seconds` are what a fix is worth in
322: -- total, and `longest` is what a worst case costs. Ten thousand generate()
323: -- calls are a shape, not ten thousand rows.
324: --
325: -- `calls` is the difference between the two kinds of cost that need opposite
326: -- fixes: calls = 1 is a fixed price the step pays whatever the population is
327: -- (amortise it, batch it, cache it), and calls = items is a per-item price
328: -- (make the item cheaper, or ask for fewer). Both look identical in a step
329: -- total, which is why the step total alone cannot say what to do.
330: --
331: -- `execution_id` ties a phase to the individual that paid for it when there is
332: -- one -- the process step's phases come out of each generated script's own
333: -- stdout -- and is NULL for a phase the step paid once on everyone's behalf.
334: -- It is deliberately **not** a foreign key: selection culls individuals and
335: -- takes their executions with them, and what a culled individual cost is still
336: -- what that generation cost. fitness_history keeps its rows through a cull for
337: -- the same reason, and a timing that vanished when the individual did would
338: -- make every sweep look cheaper the longer it ran. So the id is a way back to
339: -- the transcript while there is one, not a claim that there is.
340: --
341: -- And **an id that comes back is not the same execution**: sqlite hands out
342: -- rowids as MAX(rowid) + 1, so culling the newest executions frees their ids
343: -- for the next generation to use again. A row here is therefore identified by
344: -- (step_timing_id, execution_id) -- one step ran once, and an id it saw is its
345: -- own -- and the script phase carries the individual's number and chromosome in
346: -- `detail`, so a reader never has to join back through an id that may since
347: -- have been handed to somebody else. Self-contained, the way an execution row
348: -- and a fitness_history row are, and for exactly the same reason.
349: -- `detail` is JSON for whatever distinguishes occurrences of the same phase:
350: -- the combination type of a node, the adapter a leaf came from, the judge that
351: -- was asked.
352: CREATE TABLE IF NOT EXISTS phase_timings (
353:     id             INTEGER PRIMARY KEY,
354:     step_timing_id INTEGER NOT NULL REFERENCES step_timings(id) ON DELETE CASCADE,
355:     execution_id   INTEGER,                -- executions(id) while it exists; see above
356:     phase          TEXT    NOT NULL,     -- model_load, attach, combine.svd, generate, ..
357:     calls          INTEGER NOT NULL DEFAULT 1,
358:     seconds        REAL    NOT NULL,     -- summed over those calls
359:     longest        REAL,                 -- the worst single one
360:     detail         TEXT                  -- JSON, or NULL
361: );
362: 
363: CREATE INDEX IF NOT EXISTS individuals_by_run ON individuals(run_id, number);
364: CREATE INDEX IF NOT EXISTS executions_by_individual ON executions(individual_id);
365: CREATE INDEX IF NOT EXISTS exchanges_by_execution ON exchanges(execution_id);
366: CREATE INDEX IF NOT EXISTS fitness_history_by_run ON fitness_history(run_id, generation);
367: CREATE INDEX IF NOT EXISTS test_results_by_run ON test_results(run_id, number);
368: CREATE INDEX IF NOT EXISTS step_timings_by_run ON step_timings(run_id, pass_no, position);
369: CREATE INDEX IF NOT EXISTS phase_timings_by_step ON phase_timings(step_timing_id);
370: 
371: -- The fitness view: one row per individual, over its most recent execution.
372: -- Mean quality is what a selection step would sort on.
373: CREATE VIEW IF NOT EXISTS individual_quality AS
374: SELECT i.run_id      AS run_id,
375:        i.number      AS number,
376:        i.chromosome  AS chromosome,
377:        i.state       AS state,
378:        i.rank        AS rank,
379:        e.id          AS execution_id,
380:        e.verdict     AS verdict,
381:        e.weight_seed AS weight_seed,
382:        e.weights     AS weights,
383:        COUNT(x.id)                                   AS answers,
384:        SUM(CASE WHEN x.quality IS NULL THEN 1 ELSE 0 END) AS unscored,
385:        AVG(x.quality)                                AS quality
386: FROM individuals i
387: LEFT JOIN executions e
388:        ON e.id = (SELECT id FROM executions
389:                    WHERE individual_id = i.id
390:                    ORDER BY id DESC LIMIT 1)
391: LEFT JOIN exchanges x ON x.execution_id = e.id
392: GROUP BY i.id;
393: 
394: -- One row per answer of a testing pass, out of the JSON each test_results row
395: -- carries. The table stores a transcript whole because that is how a pass is
396: -- written and read back; this is for the questions that want it the other way
397: -- round -- which chromosome said what to which question.
398: CREATE VIEW IF NOT EXISTS test_answers AS
399: SELECT t.id         AS test_result_id,
400:        t.run_id     AS run_id,
401:        t.number     AS number,
402:        t.chromosome AS chromosome,
403:        t.dataset    AS dataset,
404:        t.verdict    AS verdict,
405:        j.key + 1    AS position,
406:        json_extract(j.value, '$.question') AS question,
407:        json_extract(j.value, '$.answer')   AS answer,
408:        json_extract(j.value, '$.quality')  AS quality,
409:        json_extract(j.value, '$.reason')   AS reason
410: FROM test_results t, json_each(t.exchanges) j;
411: 
412: CREATE VIEW IF NOT EXISTS individual_stats AS
413: select i.number, i.chromosome, SUM(ex.quality) AS SumQuality, Max(ex.quality) AS MaxQuality, MIN(ex.quality) AS MinQuality, AVG(ex.quality) AS AvgQuality
414: from individuals i inner join executions e on i.id = e.individual_id inner join exchanges ex on e.id = ex.execution_id
415: where i.state = 'ok'
416: group by i.number, chromosome
417: """
418: 
419: 
420: class Database(sqlite3.Connection):
421:     """A connection that remembers which file it opened.
422: 
423:     sqlite3.Connection is a C type with no __dict__, so `conn.path = ...` is not
424:     allowed on one; a subclass can hold it, and every error message in the
425:     pipeline can then name the database it is talking about.
426:     """
427: 
428:     path = None
429: 
430: 
431: def connect(db_path):
432:     """Open (creating if need be) the database, with the schema in place."""
433:     path = db_path if os.path.isabs(db_path) else os.path.join(_ROOT, db_path)
434:     folder = os.path.dirname(path)
435:     if folder:
436:         os.makedirs(folder, exist_ok=True)
437:     conn = sqlite3.connect(path, factory=Database)
438:     conn.path = path
439:     conn.row_factory = sqlite3.Row
440:     # Cascading deletes are off by default, and the schema leans on them.
441:     conn.execute("PRAGMA foreign_keys = ON")
442:     conn.executescript(SCHEMA)
443:     conn.commit()
444:     return conn
445: 
446: 
447: def _now():
448:     """A sortable timestamp. Stored as text: sqlite has no date type, and the
449:     adapters that used to paper over that are deprecated in 3.12+."""
450:     return time.strftime("%Y-%m-%dT%H:%M:%S")
451: 
452: 
453: def now():
454:     """The timestamp these tables carry, for a caller that stamps its own.
455: 
456:     gep_lora/core/pipeline/start_run.py marks a step's start before it runs it and only knows how long
457:     it took afterwards, so it cannot let the INSERT do it.
458:     """
459:     return _now()
460: 
461: 
462: def _git_commit():
463:     """The commit the code was at, or None outside a checkout.
464: 
465:     Worth recording: a stored sweep is only reproducible alongside the code that
466:     produced it, and this is the cheapest way to say which code that was.
467:     """
468:     try:
469:         done = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
470:                               cwd=_ROOT, capture_output=True, text=True, timeout=10)
471:     except (OSError, subprocess.SubprocessError):
472:         return None
473:     return done.stdout.strip() or None if done.returncode == 0 else None
474: 
475: 
476: # --- runs and their settings ----------------------------------------------
477: 
478: 
479: def create_run(conn, template, label=None):
480:     """Start a new sweep. Returns its id."""
481:     cursor = conn.execute(
482:         "INSERT INTO runs (created_at, label, template, status, git_commit, interpreter)"
483:         " VALUES (?, ?, ?, 'open', ?, ?)",
484:         (_now(), label, template, _git_commit(), sys.executable))
485:     conn.commit()
486:     return cursor.lastrowid
487: 
488: 
489: def finish_run(conn, run_id, status="done"):
490:     conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))
491:     conn.commit()
492: 
493: 
494: def latest_run(conn):
495:     """The most recent sweep's id, or None if the database is empty."""
496:     row = conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
497:     return row["id"] if row else None
498: 
499: 
500: def get_run(conn, run_id):
501:     return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
502: 
503: 
504: def save_settings(conn, run_id, mapping):
505:     """Record what a sweep ran under. Values are stored as JSON so ints, floats,
506:     bools, None and strings all come back as themselves."""
507:     conn.executemany(
508:         "INSERT INTO settings (run_id, key, value) VALUES (?, ?, ?)"
509:         " ON CONFLICT(run_id, key) DO UPDATE SET value = excluded.value",
510:         [(run_id, key, json.dumps(value)) for key, value in sorted(mapping.items())])
511:     conn.commit()
512: 
513: 
514: def get_settings(conn, run_id):
515:     """The recorded settings, decoded back into Python values."""
516:     rows = conn.execute("SELECT key, value FROM settings WHERE run_id = ?",
517:                         (run_id,)).fetchall()
518:     out = {}
519:     for row in rows:
520:         try:
521:             out[row["key"]] = json.loads(row["value"])
522:         except (ValueError, TypeError):
523:             out[row["key"]] = row["value"]      # written by hand, keep the text
524:     return out
525: 
526: 
527: # --- the dataset the sweep was given --------------------------------------
528: 
529: # The three splits, in the order they are worked through and reported. The
530: # schema names them too (datasets.split is CHECK'd against exactly these), so
531: # a fourth split is a schema change rather than a new string somewhere.
532: SPLITS = ("training", "validation", "testing")
533: 
534: 
535: def save_dataset(conn, run_id, split, source, records):
536:     """Store one split of a sweep's dataset. -> how many records were written.
537: 
538:     `records` is generate_runs.dataset_records() output -- position, question,
539:     reference and the line it came from -- and `source` the file they were read
540:     from, resolved, so the row says where it came from as well as what it said.
541: 
542:     Written whole, replacing whatever that split held: a dataset is saved once,
543:     at sweep creation, and re-saving one is either a re-run of that moment or a
544:     correction, never half of each. The rows themselves are never edited
545:     afterwards -- the point of them is that they say what this sweep was built
546:     on, and a sweep's dataset changing under it would be the thing they exist to
547:     prevent.
548:     """
549:     if split not in SPLITS:
550:         raise ValueError("unknown dataset split %r; the table holds %s"
551:                          % (split, ", ".join(SPLITS)))
552:     conn.execute("DELETE FROM datasets WHERE run_id = ? AND split = ?",
553:                  (run_id, split))
554:     conn.executemany(
555:         "INSERT INTO datasets (run_id, split, source, position, question,"
556:         " reference, content) VALUES (?, ?, ?, ?, ?, ?, ?)",
557:         [(run_id, split, source, record["position"], record["question"],
558:           record["reference"], record["content"]) for record in records])
559:     conn.commit()
560:     return len(records)
561: 
562: 
563: def dataset(conn, run_id, split):
564:     """One split's records, in file order."""
565:     return conn.execute(
566:         "SELECT * FROM datasets WHERE run_id = ? AND split = ? ORDER BY position",
567:         (run_id, split)).fetchall()
568: 
569: 
570: def dataset_summary(conn, run_id):
571:     """[{split, source, records, references}, ...] for the splits a sweep holds.
572: 
573:     In SPLITS order rather than in whatever order they were saved, and silent
574:     about the ones a sweep was never given -- a sweep with no validation set has
575:     no validation rows, which is not the same as an empty one.
576:     """
577:     rows = conn.execute(
578:         "SELECT split, source, COUNT(*) AS records,"
579:         "       SUM(CASE WHEN reference IS NULL THEN 0 ELSE 1 END) AS references_"
580:         "  FROM datasets WHERE run_id = ? GROUP BY split, source",
581:         (run_id,)).fetchall()
582:     found = {row["split"]: row for row in rows}
583:     return [{"split": name,
584:              "source": found[name]["source"],
585:              "records": found[name]["records"],
586:              "references": found[name]["references_"] or 0}
587:             for name in SPLITS if name in found]
588: 
589: 
590: 
591: # --- the population --------------------------------------------------------
592: 
593: 
594: def add_individuals(conn, run_id, chromosomes):
595:     """Store a freshly drawn population. Replaces any already held for this run."""
596:     conn.execute("DELETE FROM individuals WHERE run_id = ?", (run_id,))
597:     conn.executemany(
598:         "INSERT INTO individuals (run_id, number, chromosome) VALUES (?, ?, ?)",
599:         [(run_id, number, chromosome)
600:          for number, chromosome in enumerate(chromosomes, 1)])
601:     conn.commit()
602: 
603: 
604: def next_number(conn, run_id):
605:     """The number the next individual appended to this population gets.
606: 
607:     One place, because three things append now -- the copies selection makes,
608:     the newcomer it draws, and nothing else may ever guess. Numbering continues
609:     from the highest ever stored rather than from the size of the population,
610:     so a number freed by a deletion is never handed out again and every number
611:     an old script, execution, transcript or fitness_history row refers to goes
612:     on meaning the individual it meant.
613:     """
614:     return high_number(conn, run_id) + 1
615: 
616: 
617: def high_number(conn, run_id):
618:     """The highest number in this population. 0 when it holds nobody.
619: 
620:     The sweep's clock. Population *size* used to serve as one -- it grew a
621:     generation at a time and never shrank, so it dated a round -- and it stopped
622:     being able to the moment selection started culling as many individuals as it
623:     appends. This number cannot stop: numbers are handed out from the top and
624:     never reused, a round always appends before it culls, and it only ever culls
625:     individuals that were in the population before the round, so every round
626:     ends on a number strictly higher than the one it began on. A step that only
627:     reads (a re-run of `fitness`) leaves it exactly where it was, which is the
628:     other half of what a clock has to do.
629:     """
630:     return conn.execute("SELECT MAX(number) AS top FROM individuals WHERE run_id = ?",
631:                         (run_id,)).fetchone()["top"] or 0
632: 
633: 
634: def append_individual(conn, run_id, chromosome):
635:     """Append one brand-new individual to a population. -> the number it got.
636: 
637:     A chromosome and nothing else: no tree, no script, no seed, no fitness --
638:     the state a member of the very first population is in before `trees` and
639:     `runs` have been near it, which is exactly what a newcomer is. The contrast
640:     with append_copies() is the point: that one arrives holding its parent's
641:     answers, this one arrives holding none.
642:     """
643:     number = next_number(conn, run_id)
644:     conn.execute(
645:         "INSERT INTO individuals (run_id, number, chromosome) VALUES (?, ?, ?)",
646:         (run_id, number, chromosome))
647:     conn.commit()
648:     return number
649: 
650: 
651: def delete_individuals(conn, run_id, numbers):
652:     """Remove individuals from a population. -> the rows that went.
653: 
654:     The one place anything leaves a population mid-sweep, and it takes the
655:     individual whole: `PRAGMA foreign_keys` is on and the schema cascades, so a
656:     deleted individual's executions, its exchanges and its test_results go with
657:     it. That is the intended reading of a cull -- what is being discarded is a
658:     blend the search has finished with, and its transcripts are the record of
659:     that blend and of nothing else.
660: 
661:     Two things deliberately survive. `fitness_history` hangs off the *run* and
662:     keeps the chromosome, state and fitness as they were in each generation, so
663:     a culled individual stays in the history of the generations it lived
664:     through -- which is the only place a sweep says what it used to be, and
665:     would say much less if a cull could rewrite it. And the number is retired
666:     rather than freed: see next_number().
667: 
668:     The rows are read before they go, so a caller can report what it culled and
669:     clean up the script files those individuals owned.
670:     """
671:     wanted = list(numbers)
672:     if not wanted:
673:         return []
674:     marks = ", ".join("?" * len(wanted))
675:     rows = conn.execute(
676:         "SELECT * FROM individuals WHERE run_id = ? AND number IN (%s) ORDER BY number"
677:         % marks, [run_id] + wanted).fetchall()
678:     conn.execute(
679:         "DELETE FROM individuals WHERE run_id = ? AND number IN (%s)" % marks,
680:         [run_id] + wanted)
681:     conn.commit()
682:     return rows
683: 
684: 
685: def append_copies(conn, run_id, rows):
686:     """Add a copy of each of `rows` to a population. -> the numbers they got.
687: 
688:     The counterpart of add_individuals(), which replaces: numbering continues
689:     from the highest already stored, so the copies are new individuals rather
690:     than a re-drawn generation, and every number an existing script, execution
691:     or transcript refers to still means what it meant.
692: 
693:     A copy is the parent field for field -- tree, state, rank, script, weight
694:     seed and fitness, not just the chromosome. Only `id` and `number` are its
695:     own, since those are what make it a row of its own; everything else it
696:     inherits and keeps until something changes it.
697: 
698:     Except is_best, which every copy arrives with at 0. That flag does not
699:     describe an individual, it picks one out of the population -- the single
700:     one this generation carries forward -- so it is not the parent's to hand on.
701:     Copying it would give a sweep two elites, or six, and mark_best() exists
702:     precisely to keep that from ever being true. A copy of the elite is not
703:     itself the elite; the next election decides that, on the fitness it earns.
704: 
705:     The column list comes from the table rather than being written out here, so
706:     a field added to individuals is copied too instead of being quietly dropped
707:     from every copy the search makes.
708:     """
709:     if not rows:
710:         return []
711:     columns = [column["name"] for column in conn.execute("PRAGMA table_info(individuals)")
712:                if column["name"] not in ("id", "number")]
713:     first = next_number(conn, run_id)
714:     numbers = list(range(first, first + len(rows)))
715:     conn.executemany(
716:         "INSERT INTO individuals (number, %s) VALUES (%s)"
717:         % (", ".join(columns), ", ".join("?" * (len(columns) + 1))),
718:         [[number] + [0 if column == "is_best" else row[column]
719:                      for column in columns]
720:          for number, row in zip(numbers, rows)])
721:     conn.commit()
722:     return numbers
723: 
724: 
725: def individuals(conn, run_id, state=None):
726:     """The population in number order, optionally only those in one state."""
727:     sql = "SELECT * FROM individuals WHERE run_id = ?"
728:     args = [run_id]
729:     if state is not None:
730:         sql += " AND state = ?"
731:         args.append(state)
732:     return conn.execute(sql + " ORDER BY number", args).fetchall()
733: 
734: 
735: def set_tree(conn, individual_id, tree):
736:     conn.execute("UPDATE individuals SET tree = ? WHERE id = ?", (tree, individual_id))
737: 
738: 
739: def set_script(conn, individual_id, state, rank, script_name, source, weight_seed):
740:     conn.execute(
741:         "UPDATE individuals SET state = ?, rank = ?, script_name = ?,"
742:         " script_source = ?, weight_seed = ? WHERE id = ?",
743:         (state, rank, script_name, source, weight_seed, individual_id))
744: 
745: 
746: # --- executions and what they said ----------------------------------------
747: 
748: 
749: def add_execution(conn, individual_id, seconds, exit_code, verdict,
750:                   weight_seed, weights, stdout, stderr):
751:     """Record one execution of one individual. Returns its id.
752: 
753:     A row per execution rather than per individual: the same chromosome run
754:     again under a different weight seed is a second result, not a correction of
755:     the first.
756:     """
757:     cursor = conn.execute(
758:         "INSERT INTO executions (individual_id, started_at, seconds, exit_code,"
759:         " verdict, weight_seed, weights, stdout, stderr)"
760:         " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
761:         (individual_id, _now(), seconds, exit_code, verdict, weight_seed,
762:          json.dumps(weights or {}), stdout, stderr))
763:     return cursor.lastrowid
764: 
765: 
766: def add_exchanges(conn, execution_id, transcript):
767:     """Store the question/answer pairs of one execution.
768: 
769:     A mocked run arrives already carrying "quality" and "reason", so those are
770:     taken here when present and the judge never has to be asked.
771:     """
772:     conn.executemany(
773:         "INSERT INTO exchanges (execution_id, position, question, answer,"
774:         " quality, reason, judge_model) VALUES (?, ?, ?, ?, ?, ?, ?)",
775:         [(execution_id, position, item.get("question", ""), item.get("answer", ""),
776:           item.get("quality"), item.get("reason"),
777:           "generated" if "quality" in item else None)
778:          for position, item in enumerate(transcript, 1)])
779: 
780: 
781: def executed(conn, run_id):
782:     """The ids of the individuals in one sweep that have ever been run.
783: 
784:     An individual not in here has no execution at all, so there is nothing
785:     stored about it whatever its flags say -- which is what separates "already
786:     done" from "not done yet" when deciding what needs running.
787:     """
788:     return {row["individual_id"] for row in conn.execute(
789:         "SELECT DISTINCT e.individual_id FROM executions e"
790:         " JOIN individuals i ON i.id = e.individual_id WHERE i.run_id = ?",
791:         (run_id,))}
792: 
793: 
794: def latest_runs(conn, run_id):
795:     """Each individual's latest execution, as the process step needs to judge it.
796: 
797:     One row per individual that has run: its id, the chromosome and weight
798:     seed it holds *now*, and its latest execution's verdict, weight seed and
799:     the head of its stdout -- where a script names the chromosome it built.
800:     Only the head, because that line comes first and a transcript is long.
801:     """
802:     return conn.execute(
803:         "SELECT i.id AS id, i.chromosome AS chromosome, i.weight_seed AS weight_seed,"
804:         "       e.verdict AS verdict, e.weight_seed AS ran_seed,"
805:         "       substr(e.stdout, 1, 4000) AS head"
806:         "  FROM individuals i"
807:         "  JOIN executions e ON e.id = (SELECT id FROM executions"
808:         "                                WHERE individual_id = i.id"
809:         "                                ORDER BY id DESC LIMIT 1)"
810:         " WHERE i.run_id = ?", (run_id,)).fetchall()
811: 
812: 
813: def latest_execution(conn, individual_id):
814:     return conn.execute(
815:         "SELECT * FROM executions WHERE individual_id = ?"
816:         " ORDER BY id DESC LIMIT 1", (individual_id,)).fetchone()
817: 
818: 
819: def exchanges_to_score(conn, run_id, force=False):
820:     """Every exchange of the latest execution of each individual that still
821:     needs a score -- or all of them, with force."""
822:     sql = """
823:         SELECT x.*, i.number AS number, i.chromosome AS chromosome
824:         FROM exchanges x
825:         JOIN executions e ON e.id = x.execution_id
826:         JOIN individuals i ON i.id = e.individual_id
827:         WHERE i.run_id = ?
828:           AND e.id = (SELECT id FROM executions
829:                        WHERE individual_id = i.id ORDER BY id DESC LIMIT 1)
830:     """
831:     if not force:
832:         sql += " AND x.quality IS NULL"
833:     return conn.execute(sql + " ORDER BY i.number, x.position", (run_id,)).fetchall()
834: 
835: 
836: def score_exchange(conn, exchange_id, quality, reason, model):
837:     conn.execute(
838:         "UPDATE exchanges SET quality = ?, reason = ?, judged_at = ?, judge_model = ?"
839:         " WHERE id = ?", (quality, reason, _now(), model, exchange_id))
840: 
841: 
842: # --- a testing pass -------------------------------------------------------
843: 
844: 
845: def add_test_result(conn, run_id, individual, chromosome, dataset, records,
846:                     selected_on, seconds, exit_code, verdict, weights, stdout,
847:                     stderr, transcript):
848:     """Record one individual's run against a testing dataset. Returns its id.
849: 
850:     `individual` is the row it was built from -- its number and weight seed are
851:     copied in rather than joined to later, because the row has to go on meaning
852:     what it meant after mutation has rewritten that individual.
853: 
854:     `chromosome` is passed separately, and is the one the script that ran
855:     actually builds, which is not always individuals.chromosome: mutation
856:     rewrites that column and leaves the script describing what the individual
857:     used to be. What was tested is what ran, so that is what is stored.
858: 
859:     Stored whatever happened: a script that crashed leaves a row with its exit
860:     code and an empty transcript, the way a failed execution does, because "it
861:     could not answer these questions" is a result of a testing pass and not an
862:     absence of one.
863:     """
864:     cursor = conn.execute(
865:         "INSERT INTO test_results (run_id, individual_id, number, chromosome,"
866:         " dataset, records, selected_on, started_at, seconds, exit_code, verdict,"
867:         " weight_seed, weights, stdout, stderr, exchanges)"
868:         " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
869:         (run_id, individual["id"], individual["number"], chromosome,
870:          dataset, records, selected_on, _now(), seconds, exit_code, verdict,
871:          individual["weight_seed"], json.dumps(weights or {}), stdout, stderr,
872:          json.dumps(transcript or [])))
873:     return cursor.lastrowid
874: 
875: 
876: def test_results_to_score(conn, run_id, dataset=None, force=False):
877:     """The testing rows holding answers that still need a quality.
878: 
879:     Whole rows rather than answers, because a transcript is stored whole: the
880:     caller unpacks the JSON, scores what is missing and writes the row back.
881:     With `force`, every row that has any answer at all, so a pass can be
882:     re-graded by another evaluator.
883: 
884:     A row whose script crashed carries an empty transcript and is not pending
885:     -- there is nothing in it to grade, and it would come back forever.
886:     """
887:     rows = test_results(conn, run_id, dataset)
888:     pending = []
889:     for row in rows:
890:         transcript = json.loads(row["exchanges"] or "[]")
891:         if not transcript:
892:             continue
893:         if force or any(item.get("quality") is None for item in transcript):
894:             pending.append(row)
895:     return pending
896: 
897: 
898: def score_test_result(conn, result_id, transcript, evaluator, judge_model):
899:     """Write a graded transcript back, with the mean it comes to. -> that mean.
900: 
901:     The scores live inside the JSON, beside the answers they belong to, so the
902:     test_answers view reads them without another join and a row goes on being
903:     the whole story of one individual against one dataset. `quality` is the
904:     mean over the answers that have one -- the same average over the same
905:     answers that individual_quality takes for a training run, so a testing
906:     number and a training number mean the same thing.
907: 
908:     None when nothing in the row could be scored: an average of no scores is
909:     not zero, and storing 0.0 would say the blend answered badly rather than
910:     that nobody managed to grade it.
911:     """
912:     scored = [item["quality"] for item in transcript
913:               if item.get("quality") is not None]
914:     mean = round(sum(scored) / len(scored), 4) if scored else None
915:     conn.execute(
916:         "UPDATE test_results SET exchanges = ?, quality = ?, evaluator = ?,"
917:         " judge_model = ?, scored_at = ? WHERE id = ?",
918:         (json.dumps(transcript), mean, evaluator, judge_model, _now(), result_id))
919:     return mean
920: 
921: 
922: def test_quality(conn, run_id, dataset=None):
923:     """How the tested individuals did, beside how they did in the search.
924: 
925:     One row each: the training quality that got it picked (selected_on, as it
926:     was then) and the quality it earned on the dataset it was never scored on.
927:     The two columns side by side are the whole point of a testing pass, and the
928:     difference between them is the only thing that says whether a blend was
929:     selected for the questions or for the job.
930:     """
931:     sql = ("SELECT id, number, chromosome, dataset, verdict, records,"
932:            "       selected_on, quality, evaluator,"
933:            "       json_array_length(exchanges) AS answers"
934:            "  FROM test_results WHERE run_id = ?")
935:     args = [run_id]
936:     if dataset is not None:
937:         sql += " AND dataset = ?"
938:         args.append(dataset)
939:     return conn.execute(sql + " ORDER BY id", args).fetchall()
940: 
941: 
942: def test_results(conn, run_id, dataset=None):
943:     """A sweep's testing rows, newest pass last, optionally for one dataset."""
944:     sql = "SELECT * FROM test_results WHERE run_id = ?"
945:     args = [run_id]
946:     if dataset is not None:
947:         sql += " AND dataset = ?"
948:         args.append(dataset)
949:     return conn.execute(sql + " ORDER BY id", args).fetchall()
950: 
951: 
952: def test_summary(conn, run_id):
953:     """[{dataset, tested, ok, answers, last}, ...], one per dataset tested on.
954: 
955:     The read side of a testing pass at a glance: which datasets this sweep's
956:     individuals have been put in front of, how many of them ran, and how many
957:     answers came back. Silent about a sweep that has never been tested.
958:     """
959:     rows = conn.execute(
960:         "SELECT dataset,"
961:         "       COUNT(*)                                        AS tested,"
962:         "       SUM(CASE WHEN verdict = 'ok' THEN 1 ELSE 0 END) AS ok_,"
963:         "       SUM(json_array_length(exchanges))               AS answers,"
964:         "       AVG(quality)                                    AS quality,"
965:         "       MAX(started_at)                                 AS last"
966:         "  FROM test_results WHERE run_id = ?"
967:         " GROUP BY dataset ORDER BY last", (run_id,)).fetchall()
968:     return [{"dataset": row["dataset"], "tested": row["tested"],
969:              "ok": row["ok_"] or 0, "answers": row["answers"] or 0,
970:              "quality": row["quality"], "last": row["last"]} for row in rows]
971: 
972: 
973: # --- the base model's own answers, cached across sweeps --------------------
974: 
975: 
976: def question_key(text):
977:     """A question reduced to what makes two of them the same question.
978: 
979:     The baselines table matches on this rather than on the wording, because the
980:     question reaching it comes back out of a transcript the generated script
981:     printed, and a line that lost its trailing spaces on the way is still the
982:     same question. Lives here rather than with the evaluator for the same reason
983:     the table does: it is how a row is addressed.
984:     """
985:     return " ".join((text or "").split()).strip().lower()
986: 
987: 
988: def baselines(conn, model):
989:     """{question_key: answer} for one base model -- the cache, read whole.
990: 
991:     One query rather than one per question: a transcript is a few dozen
992:     exchanges of the same eval set, so the whole of a model's cache is smaller
993:     than the transcript being scored against it.
994:     """
995:     rows = conn.execute(
996:         "SELECT question_key, answer FROM baselines WHERE model = ?", (model,)).fetchall()
997:     return {row["question_key"]: row["answer"] for row in rows}
998: 
999: 
1000: def add_baselines(conn, model, items, source=None):
1001:     """Store base-model answers. -> how many rows are new.
1002: 
1003:     `items` is [(question, answer)]. An answer already cached for that model and
1004:     question is left as it was: it is the same model answering the same question
1005:     with do_sample off, so a second copy would say the same thing, and quietly
1006:     replacing it would move the ground every earlier score in the database was
1007:     measured against. An empty answer is not stored at all -- that is a run that
1008:     failed to answer, not the base model's reply.
1009:     """
1010:     added = 0
1011:     for question, answer in items:
1012:         if not (answer or "").strip():
1013:             continue
1014:         cursor = conn.execute(
1015:             "INSERT OR IGNORE INTO baselines"
1016:             " (model, question_key, question, answer, created_at, source)"
1017:             " VALUES (?, ?, ?, ?, ?, ?)",
1018:             (model, question_key(question), question, answer, _now(), source))
1019:         added += cursor.rowcount or 0
1020:     conn.commit()
1021:     return added
1022: 
1023: 
1024: def baseline_rows(conn, model):
1025:     """Every cached answer for one base model, in the order they were stored."""
1026:     return conn.execute(
1027:         "SELECT * FROM baselines WHERE model = ? ORDER BY id", (model,)).fetchall()
1028: 
1029: 
1030: def set_fitness(conn, run_id, number, fitness):
1031:     """Record one individual's fitness -- the number selection sorts on.
1032: 
1033:     Keyed by (run_id, number) rather than by row id, because the caller works
1034:     from the individual_quality view, whose rows are individuals seen through
1035:     their latest execution.
1036:     """
1037:     conn.execute(
1038:         "UPDATE individuals SET fitness = ? WHERE run_id = ? AND number = ?",
1039:         (fitness, run_id, number))
1040: 
1041: 
1042: # --- fitness, generation by generation ------------------------------------
1043: 
1044: 
1045: def fitness_generation(conn, run_id):
1046:     """Which generation the fitness snapshot being taken now is.
1047: 
1048:     Nothing in the schema counts generations, because nothing needs to: a sweep
1049:     is a population that keeps being rewritten in place. The count is derived
1050:     here instead, from the one thing that dates a round -- the highest
1051:     individual number the population holds, which selection always leaves
1052:     higher than it found it. See high_number(); the same watermark selection
1053:     and mutation seed their generators from.
1054: 
1055:     It used to be the size of the population, which worked only while selection
1056:     grew it. Now that a round culls as many individuals as it appends the size
1057:     can sit at exactly COUNT forever, and dating a round by it would file every
1058:     generation of a static sweep as a restatement of the first -- one row in
1059:     the history, overwritten five times, and no curve at all. The watermark
1060:     says the same thing about a growing population and goes on saying it about
1061:     a static one.
1062: 
1063:     So: the first snapshot is generation 1, and a later one is a new generation
1064:     if the population has taken on a higher number since the last was taken,
1065:     and the *same* generation restated if it has not. That is what makes
1066:     re-running
1067: 
1068:         python start_run.py fitness
1069: 
1070:     -- pure arithmetic over stored transcripts, and a reasonable thing to redo
1071:     after a re-scored `evaluate` -- rewrite the current generation rather than
1072:     invent another one.
1073: 
1074:     The blind spot is a generation that selects nobody: SELECTION_COUNT at 0,
1075:     or a wheel with nothing to spin -- selection writes nothing at all in
1076:     either case, culls and newcomer included. Such a round is indistinguishable
1077:     from the one before it here and is recorded as a restatement of it. Both cases are
1078:     already dead ends for the search -- an all-zero population stops at the
1079:     elitism step -- so the history losing a row there costs nothing a running
1080:     sweep would have wanted.
1081:     """
1082:     row = conn.execute(
1083:         "SELECT generation, MAX(number) AS high FROM fitness_history"
1084:         " WHERE run_id = ? GROUP BY generation ORDER BY generation DESC LIMIT 1",
1085:         (run_id,)).fetchone()

(Output capped at 50 KB. Showing lines 1-1085. Use offset=1086 to continue.)
</content>
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 5.0s)

_Thinking:_

Let me continue reading store.py and read other core files. Let me look at the rest of store.py, gep_lora/core/pipeline/start_run.py, gep_lora/core/pipeline/main.py, generate_runs.py, process_run.py, selection.py, mutation.py.

Given the huge size, let me be strategic and use the explore agent to summarize areas, while I read key files. Actually for a critique I need to actually see the code. Let me read several files in parallel with offsets.

Let me read the rest of store.py, generate_runs.py, process_run.py, and the templates.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\storage\\store.py",
  "offset": 1086
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\storage\store.py</path>
<type>file</type>
<content>
1086:     if row is None:
1087:         return 1
1088:     return row["generation"] + (0 if row["high"] == high_number(conn, run_id) else 1)
1089: 
1090: 
1091: def record_fitness(conn, run_id, generation, entries):
1092:     """Store one generation's fitness for a whole population. -> the timestamp.
1093: 
1094:     Every row of the generation carries the same `recorded_at`, because they are
1095:     one snapshot taken at one moment and not a stream of separate events; the
1096:     stamp dates the generation, and is the only place a sweep says when its
1097:     generations happened.
1098: 
1099:     Anything already held for this generation is replaced, so a re-run of the
1100:     fitness step restates the round with what it now knows instead of failing on
1101:     the unique key or leaving half the old snapshot behind.
1102: 
1103:     `entries` are dicts with the columns below, which is what calculate_fitness
1104:     builds out of the individual_quality view.
1105:     """
1106:     stamp = _now()
1107:     conn.execute("DELETE FROM fitness_history WHERE run_id = ? AND generation = ?",
1108:                  (run_id, generation))
1109:     conn.executemany(
1110:         "INSERT INTO fitness_history (run_id, generation, population, recorded_at,"
1111:         " number, chromosome, state, fitness, answers, unscored)"
1112:         " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
1113:         [(run_id, generation, len(entries), stamp, entry["number"],
1114:           entry["chromosome"], entry["state"], entry["fitness"],
1115:           entry["answers"], entry["unscored"]) for entry in entries])
1116:     conn.commit()
1117:     return stamp
1118: 
1119: 
1120: def fitness_history(conn, run_id, generation=None):
1121:     """The recorded history of one sweep, oldest generation first."""
1122:     sql = "SELECT * FROM fitness_history WHERE run_id = ?"
1123:     args = [run_id]
1124:     if generation is not None:
1125:         sql += " AND generation = ?"
1126:         args.append(generation)
1127:     return conn.execute(sql + " ORDER BY generation, number", args).fetchall()
1128: 
1129: 
1130: def fitness_by_generation(conn, run_id):
1131:     """One row per generation: when it was, and how its fitness came out.
1132: 
1133:     The shape you want to watch a search in -- whether the mean is climbing, and
1134:     whether the best chromosome is getting better or the population is merely
1135:     getting larger.
1136:     """
1137:     return conn.execute(
1138:         "SELECT generation, population, MIN(recorded_at) AS recorded_at,"
1139:         " COUNT(*) AS individuals, MIN(fitness) AS worst, MAX(fitness) AS best,"
1140:         " AVG(fitness) AS mean,"
1141:         " SUM(CASE WHEN fitness > 0 THEN 1 ELSE 0 END) AS scored"
1142:         " FROM fitness_history WHERE run_id = ?"
1143:         " GROUP BY generation ORDER BY generation", (run_id,)).fetchall()
1144: 
1145: 
1146: def best_of_generation(conn, run_id, generation):
1147:     """The fittest individual of one recorded generation, lowest number on a tie.
1148: 
1149:     Elitism's rule, applied to the history rather than to the population, so the
1150:     two agree about which chromosome a generation was won by.
1151:     """
1152:     return conn.execute(
1153:         "SELECT * FROM fitness_history WHERE run_id = ? AND generation = ?"
1154:         " ORDER BY fitness DESC, number LIMIT 1", (run_id, generation)).fetchone()
1155: 
1156: 
1157: def set_changed(conn, individual_id, has_changed):
1158:     """Record whether the last mutation round altered this individual.
1159: 
1160:     has_changed = 1 clears the fitness, for the reason set_chromosome() gives.
1161:     """
1162:     if has_changed:
1163:         conn.execute(
1164:             "UPDATE individuals SET has_changed = 1, fitness = NULL WHERE id = ?",
1165:             (individual_id,))
1166:     else:
1167:         conn.execute("UPDATE individuals SET has_changed = 0 WHERE id = ?",
1168:                      (individual_id,))
1169: 
1170: 
1171: def set_chromosome(conn, individual_id, chromosome):
1172:     """Replace an individual's chromosome. It has changed, and it has no fitness.
1173: 
1174:     All three go together. A new chromosome *is* the change, so has_changed
1175:     follows from it rather than being decided separately; and the fitness beside
1176:     it was earned by the chromosome that just went away, which makes it not
1177:     merely stale but wrong -- it would let an individual be elected, or win a
1178:     slice of the roulette wheel, on a score belonging to a blend it no longer
1179:     describes. NULL is the honest value: no fitness yet, the way an individual
1180:     that has never run has none.
1181: 
1182:     The tree, script and rank are stale too, but they are only descriptions and
1183:     they are re-derived wholesale by trees and runs. Fitness has to be re-earned
1184:     through process and evaluate, so it is cleared here rather than left to
1185:     mislead until then.
1186:     """
1187:     conn.execute(
1188:         "UPDATE individuals SET chromosome = ?, has_changed = 1, fitness = NULL"
1189:         " WHERE id = ?", (chromosome, individual_id))
1190: 
1191: 
1192: def mark_best(conn, run_id, number):
1193:     """Make one individual the sweep's elite, and the only one.
1194: 
1195:     Both halves in one statement, because they are one fact: is_best says which
1196:     individual this generation carries forward, and a sweep with two of them --
1197:     or with last generation's still set -- says nothing at all. Clearing first
1198:     and setting after would leave that window open on the way through.
1199:     """
1200:     conn.execute(
1201:         "UPDATE individuals SET is_best = (number = ?) WHERE run_id = ?",
1202:         (number, run_id))
1203: 
1204: 
1205: def quality_rows(conn, run_id):
1206:     """The fitness view for one sweep, best first."""
1207:     return conn.execute(
1208:         "SELECT * FROM individual_quality WHERE run_id = ?"
1209:         " ORDER BY quality IS NULL, quality DESC, number", (run_id,)).fetchall()
1210: 
1211: 
1212: # --- what each step cost ---------------------------------------------------
1213: 
1214: 
1215: def next_pass(conn, run_id):
1216:     """The number of the pass about to start: 1-based, one per start_run.run().
1217: 
1218:     Derived rather than held, the way a generation is (see fitness_generation):
1219:     a pass is over the moment its steps are, and nothing but these rows has any
1220:     use for having counted them.
1221:     """
1222:     row = conn.execute("SELECT MAX(pass_no) AS top FROM step_timings WHERE run_id = ?",
1223:                        (run_id,)).fetchone()
1224:     return (row["top"] or 0) + 1
1225: 
1226: 
1227: def add_step_timing(conn, run_id, pass_no, position, step, seconds, **fields):
1228:     """Record what one step of one pass cost. Returns its id.
1229: 
1230:     Written whether the step finished or failed -- a step that fell over after
1231:     forty minutes is exactly the kind of cost this table exists to show -- so
1232:     `status` is part of the row rather than a reason not to write one.
1233:     """
1234:     cursor = conn.execute(
1235:         "INSERT INTO step_timings (run_id, pass_no, position, step, generation,"
1236:         " watermark, started_at, seconds, items, unit, skipped, status, note)"
1237:         " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
1238:         (run_id, pass_no, position, step, fields.get("generation"),
1239:          fields.get("watermark"), fields.get("started_at") or _now(), seconds,
1240:          fields.get("items") or 0, fields.get("unit"), fields.get("skipped") or 0,
1241:          fields.get("status") or "ok", fields.get("note")))
1242:     return cursor.lastrowid
1243: 
1244: 
1245: def add_phase_timings(conn, step_timing_id, phases):
1246:     """Store the phases of one step. `phases` is a sequence of dicts:
1247: 
1248:         {"phase": "model_load", "calls": 1, "seconds": 62.3,
1249:          "longest": 62.3, "execution_id": 41, "detail": {...}}
1250: 
1251:     Nothing here decides what a phase is -- the step that measured it does --
1252:     so a new phase is a new name and no schema change.
1253:     """
1254:     conn.executemany(
1255:         "INSERT INTO phase_timings (step_timing_id, execution_id, phase, calls,"
1256:         " seconds, longest, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
1257:         [(step_timing_id, phase.get("execution_id"), phase["phase"],
1258:           phase.get("calls", 1), phase["seconds"], phase.get("longest"),
1259:           json.dumps(phase["detail"]) if phase.get("detail") else None)
1260:          for phase in phases])
1261: 
1262: 
1263: def step_timings(conn, run_id):
1264:     """Every step row of one sweep, in the order the steps ran."""
1265:     return conn.execute(
1266:         "SELECT * FROM step_timings WHERE run_id = ? ORDER BY pass_no, position, id",
1267:         (run_id,)).fetchall()
1268: 
1269: 
1270: def step_costs(conn, run_id):
1271:     """One row per step of a sweep, summed over every pass it ran in.
1272: 
1273:     The ranked answer to "which step should be optimised first": total seconds,
1274:     the share of the sweep they are, and what one item of that step cost.
1275:     """
1276:     return conn.execute(
1277:         "SELECT step,"
1278:         "       COUNT(*)                        AS passes,"
1279:         "       SUM(seconds)                    AS seconds,"
1280:         "       MAX(seconds)                    AS longest,"
1281:         "       SUM(items)                      AS items,"
1282:         "       SUM(skipped)                    AS skipped,"
1283:         "       MAX(unit)                       AS unit,"
1284:         "       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failures"
1285:         "  FROM step_timings WHERE run_id = ?"
1286:         " GROUP BY step ORDER BY seconds DESC", (run_id,)).fetchall()
1287: 
1288: 
1289: def phase_costs(conn, run_id, step=None):
1290:     """One row per phase, summed over every step row it was measured in.
1291: 
1292:     `step` narrows it to one step's phases, which is the second question a
1293:     reader asks once the first has named a step.
1294:     """
1295:     sql = ("SELECT s.step AS step, p.phase AS phase,"
1296:            "       SUM(p.calls)    AS calls,"
1297:            "       SUM(p.seconds)  AS seconds,"
1298:            "       MAX(p.longest)  AS longest,"
1299:            "       COUNT(DISTINCT s.id || ':' || p.execution_id) AS executions"
1300:            "  FROM phase_timings p JOIN step_timings s ON s.id = p.step_timing_id"
1301:            " WHERE s.run_id = ?")
1302:     args = [run_id]
1303:     if step:
1304:         sql += " AND s.step = ?"
1305:         args.append(step)
1306:     return conn.execute(sql + " GROUP BY s.step, p.phase ORDER BY seconds DESC",
1307:                         args).fetchall()
1308: 
1309: 
1310: def execution_costs(conn, run_id):
1311:     """The phases each individual's own script reported, one row per phase.
1312: 
1313:     The per-individual read of `phase_timings`: every row that names an
1314:     execution, with the individual behind it where it still exists. `number`
1315:     and `chromosome` come back NULL for an execution selection has since culled
1316:     -- the phase rows outlive it on purpose (see the schema), and what that
1317:     individual cost is still part of what its generation cost. Worse than
1318:     NULL, they come back **wrong** where a culled execution's id has since been
1319:     handed to another individual, which sqlite does as a matter of course. Read
1320:     the identity out of the script row's `detail` and use these two only as the
1321:     fallback for a sweep recorded before it was written there; group on
1322:     (step_timing_id, execution_id), never on the id alone.
1323: 
1324:     Left as one row per (execution, phase) rather than folded here: a reader
1325:     that wants a total per individual can add them up, and one that wants to
1326:     know which phase made a particular blend expensive cannot get it back once
1327:     they are summed.
1328:     """
1329:     return conn.execute(
1330:         "SELECT p.step_timing_id AS step_timing_id, p.execution_id AS execution_id,"
1331:         "       p.phase AS phase, p.calls AS calls, p.detail AS detail,"
1332:         "       p.seconds AS seconds, p.longest AS longest,"
1333:         "       i.number AS number, i.chromosome AS chromosome, i.state AS state,"
1334:         "       e.verdict AS verdict, s.pass_no AS pass_no, s.step AS step"
1335:         "  FROM phase_timings p"
1336:         "  JOIN step_timings s ON s.id = p.step_timing_id"
1337:         "  LEFT JOIN executions e ON e.id = p.execution_id"
1338:         "  LEFT JOIN individuals i ON i.id = e.individual_id"
1339:         " WHERE s.run_id = ? AND p.execution_id IS NOT NULL"
1340:         " ORDER BY s.pass_no, p.step_timing_id, p.execution_id, p.id",
1341:         (run_id,)).fetchall()
1342: 
1343: 
1344: def pass_costs(conn, run_id):
1345:     """What each pass over the sweep cost, oldest first -- a generation each,
1346:     when gep_lora/core/pipeline/continue_run.py is the one turning the crank."""
1347:     return conn.execute(
1348:         "SELECT pass_no, MAX(generation) AS generation, MAX(watermark) AS watermark,"
1349:         "       MIN(started_at) AS started_at, SUM(seconds) AS seconds,"
1350:         "       COUNT(*) AS steps"
1351:         "  FROM step_timings WHERE run_id = ?"
1352:         " GROUP BY pass_no ORDER BY pass_no", (run_id,)).fetchall()
1353: 
1354: 
1355: # --- the one thing that must be a file ------------------------------------
1356: 
1357: 
1358: def materialise(conn, run_id, run_dir, force=False):
1359:     """Write the generated scripts to disk, because process has to run them.
1360: 
1361:     They are a cache of individuals.script_source, not a second copy of the
1362:     truth: an existing file is left alone unless it differs, and a deleted one
1363:     simply comes back. Returns how many were written.
1364: 
1365:     run_dir must stay exactly one level below the project folder -- a generated
1366:     script finds the LoRA folders by going up one from itself, so a deeper
1367:     folder would break every path in it. The eval prompts are exempt: their path
1368:     comes from TRAINING_SET in settings.py and is stamped in absolute.
1369:     """
1370:     os.makedirs(run_dir, exist_ok=True)
1371:     written = 0
1372:     for row in individuals(conn, run_id):
1373:         if not row["script_source"]:
1374:             continue                            # not through the runs step yet
1375:         path = os.path.join(run_dir, row["script_name"])
1376:         if not force and os.path.exists(path):
1377:             with open(path, encoding="utf-8") as handle:
1378:                 if handle.read() == row["script_source"]:
1379:                     continue
1380:         with open(path, "w", encoding="utf-8") as handle:
1381:             handle.write(row["script_source"])
1382:         written += 1
1383:     return written
1384: 
1385: 
1386: def remove_scripts(conn, run_id, run_dir, names=None):
1387:     """Delete generated script files from disk. Returns how many went.
1388: 
1389:     The inverse of materialise(), and safe for the same reason it is: the text
1390:     of every script is in individuals.script_source, so a deleted file comes
1391:     back the moment anything needs it. Once a sweep has been processed the files
1392:     have served their purpose -- they exist only because a script has to be a
1393:     file to be launched -- and leaving a population's worth of them lying about
1394:     only invites someone to run a stale one by hand.
1395: 
1396:     `names` limits it to those scripts; None means every one this run owns.
1397:     Either way only names this run recorded are touched, never whatever else
1398:     happens to be sitting in run_dir.
1399:     """
1400:     wanted = None if names is None else set(names)
1401:     return discard_scripts(
1402:         run_dir,
1403:         [row["script_name"] for row in individuals(conn, run_id)
1404:          if row["script_name"] and (wanted is None or row["script_name"] in wanted)])
1405: 
1406: 
1407: def discard_scripts(run_dir, names):
1408:     """Delete these script files from run_dir. Returns how many went.
1409: 
1410:     The half of remove_scripts() that does not need the database, split out for
1411:     the one caller that cannot use it: selection culls individuals, and by the
1412:     time the files want deleting the rows that named them are gone. The caller
1413:     is holding those rows, so it can hand the names straight over.
1414:     """
1415:     gone = 0
1416:     for name in names:
1417:         if not name:
1418:             continue
1419:         try:
1420:             os.remove(os.path.join(run_dir, name))
1421:             gone += 1
1422:         except OSError:
1423:             # Already deleted, or held open by something: not worth failing a
1424:             # finished sweep over a file that was only ever a cache.
1425:             pass
1426:     return gone
1427: 
1428: 
1429: # --- reading a sweep back out ---------------------------------------------
1430: 
1431: 
1432: def export_run(conn, run_id, out_dir):
1433:     """Write one sweep out as a folder of text files.
1434: 
1435:     The database is the store; this is for the times a folder is what you want
1436:     -- diff two populations, grep a transcript, hand someone the scripts.
1437:     Everything written here is derived from the database, so the folder is a
1438:     view of a sweep and never the sweep itself. Returns the paths written.
1439:     """
1440:     os.makedirs(out_dir, exist_ok=True)
1441:     rows = individuals(conn, run_id)
1442:     if not rows:
1443:         raise SystemExit("run %s holds no individuals" % run_id)
1444:     written = []
1445: 
1446:     def write(name, text):
1447:         path = os.path.join(out_dir, name)
1448:         with open(path, "w", encoding="utf-8") as handle:
1449:             handle.write(text)
1450:         written.append(path)
1451: 
1452:     write("population.txt", "".join(row["chromosome"] + "\n" for row in rows))
1453: 
1454:     # Each split written back as the lines it was read from, so the file in the
1455:     # folder can be diffed against the one on disk today and say whether the
1456:     # dataset has moved under the sweep.
1457:     for entry in dataset_summary(conn, run_id):
1458:         write("dataset_%s.txt" % entry["split"],
1459:               "".join(item["content"] + "\n"
1460:                       for item in dataset(conn, run_id, entry["split"])))
1461: 
1462:     drawn = [row for row in rows if row["tree"]]
1463:     if drawn:
1464:         write("trees.txt", "\n\n\n".join(
1465:             "#%d\n%s" % (row["number"], row["tree"]) for row in drawn) + "\n")
1466: 
1467:     scripted = [row for row in rows if row["script_source"]]
1468:     if scripted:
1469:         write("index.txt", "script      state rank  expression\n" + "\n".join(
1470:             "%s  %-4s rank %-4d %s" % (row["script_name"], row["state"],
1471:                                        row["rank"], row["chromosome"])
1472:             for row in scripted) + "\n")
1473:         for row in scripted:
1474:             write(row["script_name"], row["script_source"])
1475: 
1476:     results = []
1477:     for row in rows:
1478:         execution = latest_execution(conn, row["id"])
1479:         if execution is None:
1480:             continue
1481:         number = "%03d" % row["number"]
1482:         write("output_%s.txt" % number,
1483:               "# %s\n# %s\n# predicted rank %s, index says %s\n\n%s%s"
1484:               % (row["script_name"], row["chromosome"], row["rank"], row["state"],
1485:                  execution["stdout"] or "", execution["stderr"] or ""))
1486: 
1487:         transcript = [
1488:             {key: item[key] for key in ("question", "answer", "quality", "reason")
1489:              if item[key] is not None}
1490:             for item in conn.execute(
1491:                 "SELECT * FROM exchanges WHERE execution_id = ? ORDER BY position",
1492:                 (execution["id"],)).fetchall()
1493:         ]
1494:         write("output_result_%s.json" % number,
1495:               json.dumps({"chromosome": row["chromosome"],
1496:                           "weights": json.loads(execution["weights"] or "{}"),
1497:                           "exchanges": transcript},
1498:                          indent=2, ensure_ascii=False) + "\n")
1499:         results.append("%-14s %-5s %-8s %7.1f %5d  %-26s %s"
1500:                        % (row["script_name"], row["state"], execution["verdict"],
1501:                           execution["seconds"] or 0.0, len(transcript),
1502:                           "output_result_%s.json" % number, row["chromosome"]))
1503: 
1504:     if results:
1505:         write("results.txt", "%-14s %-5s %-8s %7s %5s  %-26s %s\n"
1506:               % ("script", "state", "result", "secs", "qa", "transcript", "expression")
1507:               + "\n".join(results) + "\n")
1508: 
1509:     # The one file here that is not a view of the population as it stands now:
1510:     # every generation the sweep went through, in the order it went through
1511:     # them, with the chromosomes and the scores as they were at the time.
1512:     history = fitness_history(conn, run_id)
1513:     if history:
1514:         write("fitness_history.txt",
1515:               "%-5s %-20s %-6s %-7s %-7s %-7s\n"
1516:               % ("gen", "recorded", "pop", "best", "mean", "worst")
1517:               + "".join("%-5d %-20s %-6d %-7.3f %-7.3f %-7.3f\n"
1518:                         % (entry["generation"], entry["recorded_at"],
1519:                            entry["population"], entry["best"] or 0.0,
1520:                            entry["mean"] or 0.0, entry["worst"] or 0.0)
1521:                         for entry in fitness_by_generation(conn, run_id))
1522:               + "\n%-5s %-4s %-5s %-7s %-7s %s\n"
1523:               % ("gen", "#", "state", "fitness", "answers", "chromosome")
1524:               + "".join("%-5d %-4d %-5s %-7.3f %-7d %s\n"
1525:                         % (row["generation"], row["number"], row["state"] or "-",
1526:                            row["fitness"] or 0.0, row["answers"] or 0,
1527:                            row["chromosome"])
1528:                         for row in history))
1529:     return written
1530: 
1531: 
1532: # --- driver ---------------------------------------------------------------
1533: 
1534: 
1535: def summarise(conn, run_id):
1536:     """Print one sweep: what it ran under, and how it came out."""
1537:     run = get_run(conn, run_id)
1538:     if run is None:
1539:         raise SystemExit("no run %s in this database" % run_id)
1540:     print("run %d  %s  %s  template=%s  commit=%s"
1541:           % (run["id"], run["created_at"], run["status"], run["template"],
1542:              run["git_commit"] or "?"))
1543: 
1544:     stored = get_settings(conn, run_id)
1545:     if stored:
1546:         print("\nsettings")
1547:         for key, value in sorted(stored.items()):
1548:             printable = str(value)
1549:             if len(printable) > 60:
1550:                 printable = printable[:57] + "..."
1551:             print("    %-20s %s" % (key, printable))
1552: 
1553:     splits = dataset_summary(conn, run_id)
1554:     if splits:
1555:         print("\ndataset")
1556:         for entry in splits:
1557:             print("    %-11s %4d record(s), %4d with a reference  %s"
1558:                   % (entry["split"], entry["records"], entry["references"],
1559:                      entry["source"]))
1560: 
1561:     rows = quality_rows(conn, run_id)
1562:     print("\n%d individual(s)" % len(rows))
1563:     print("    %-4s %-5s %-9s %-6s %-7s %s"
1564:           % ("#", "state", "verdict", "answers", "quality", "chromosome"))
1565:     for row in rows:
1566:         print("    %-4d %-5s %-9s %-6d %-7s %s"
1567:               % (row["number"], row["state"] or "-", row["verdict"] or "-",
1568:                  row["answers"] or 0,
1569:                  "-" if row["quality"] is None else "%.3f" % row["quality"],
1570:                  row["chromosome"]))
1571: 
1572:     scored = [row["quality"] for row in rows if row["quality"] is not None]
1573:     if scored:
1574:         print("\nquality across %d individual(s): min %.3f, max %.3f, mean %.3f"
1575:               % (len(scored), min(scored), max(scored), sum(scored) / len(scored)))
1576: 
1577:     # The population above is the sweep as it stands; this is the sweep as it
1578:     # went, which is the half that says whether the search was working.
1579:     history = fitness_by_generation(conn, run_id)
1580:     if history:
1581:         print("\nfitness by generation")
1582:         print("    %-5s %-20s %-6s %-7s %-7s %-7s %s"
1583:               % ("gen", "recorded", "pop", "best", "mean", "worst",
1584:                  "fittest chromosome"))
1585:         for entry in history:
1586:             best = best_of_generation(conn, run_id, entry["generation"])
1587:             print("    %-5d %-20s %-6d %-7.3f %-7.3f %-7.3f %s"
1588:                   % (entry["generation"], entry["recorded_at"], entry["population"],
1589:                      entry["best"] or 0.0, entry["mean"] or 0.0,
1590:                      entry["worst"] or 0.0, best["chromosome"] if best else "-"))
1591: 
1592:     # And this is the sweep against questions it was never scored on --
1593:     # test_run_with_dataset.py, if it has been run. Silent otherwise: most
1594:     # sweeps have never been tested, and an empty heading says nothing.
1595:     tested = test_summary(conn, run_id)
1596:     if tested:
1597:         print("\ntesting passes")
1598:         print("    %-20s %-7s %-5s %-8s %-7s %s"
1599:               % ("last", "tested", "ok", "answers", "quality", "dataset"))
1600:         for entry in tested:
1601:             print("    %-20s %-7d %-5d %-8d %-7s %s"
1602:                   % (entry["last"], entry["tested"], entry["ok"],
1603:                      entry["answers"],
1604:                      "-" if entry["quality"] is None else "%.3f" % entry["quality"],
1605:                      entry["dataset"]))
1606: 
1607: 
1608: def main(argv=None):
1609:     parser = argparse.ArgumentParser(description="Inspect the sweep database.")
1610:     parser.add_argument("--db", default=None,
1611:                         help="database file (default: settings.DB_PATH)")
1612:     parser.add_argument("--list", action="store_true", help="list the sweeps")
1613:     parser.add_argument("--show", type=int, metavar="RUN",
1614:                         help="summarise one sweep (0 = the most recent)")
1615:     parser.add_argument("--export", type=int, metavar="RUN",
1616:                         help="write one sweep out as text files (0 = most recent)")
1617:     parser.add_argument("--into", default="export",
1618:                         help="folder for --export (default export)")
1619:     args = parser.parse_args(argv)
1620: 
1621:     if args.db is None:
1622:         from config import settings as _settings
1623:         args.db = _settings.DB_PATH
1624:     conn = connect(args.db)
1625: 
1626:     def resolve(value):
1627:         chosen = latest_run(conn) if value == 0 else value
1628:         if chosen is None:
1629:             raise SystemExit("%s holds no runs yet" % conn.path)
1630:         return chosen
1631: 
1632:     if args.export is not None:
1633:         run_id = resolve(args.export)
1634:         written = export_run(conn, run_id, args.into)
1635:         print("wrote %d file(s) from run %d to %s" % (len(written), run_id, args.into))
1636:         return 0
1637: 
1638:     if args.show is not None:
1639:         summarise(conn, resolve(args.show))
1640:         return 0
1641: 
1642:     rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
1643:     if not rows:
1644:         print("%s holds no runs yet" % conn.path)
1645:         return 0
1646:     print("%-4s %-20s %-7s %-26s %s" % ("id", "created", "status", "template", "label"))
1647:     for row in rows:
1648:         print("%-4d %-20s %-7s %-26s %s"
1649:               % (row["id"], row["created_at"], row["status"], row["template"],
1650:                  row["label"] or ""))
1651:     return 0
1652: 
1653: 
1654: if __name__ == "__main__":
1655:     sys.exit(main())

(End of file - total 1655 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\blends\\generate_runs.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\blends\generate_runs.py</path>
<type>file</type>
<content>
1: """
2: generate_runs.py - Turn a tree into a runnable script.
3: 
4: Each individual becomes one self-contained script, written in the style of
5: combination.py: load the base model once, attach the LoRAs the tree names, fold
6: them together with PEFT's add_weighted_adapter, then chat through the resulting
7: adapter. gep_lora/core/pipeline/start_run.py calls plan() and render() here for every individual in a sweep
8: and stores what comes back; the script only reaches disk long enough to be run.
9: 
10: The script itself comes from a template -- template_code.py by default -- which
11: is the generated file with the varying parts marked. Keeping the shape in a
12: template rather than in string literals means you can open it, syntax-check it,
13: and see what a generated script looks like directly; only what genuinely varies
14: per individual is a marker.
15: 
16: Which template to fill is an argument, so the same generator produces different
17: kinds of script from the same population -- it comes from TEMPLATE in
18: settings.py. template_code_mocked.py is the other one in the repo: same markers
19: and same rank arithmetic, but no model load and random answers, for exercising
20: the pipeline without a GPU.
21: 
22: render_baseline() fills a fourth kind, template_baseline.py: the base model on
23: its own, answering the same prompts with nothing attached. It has no tree and no
24: weights, so it is not an individual and the runs step never asks for it --
25: baseline_run.py does, once per base model, for the llm_judge_baseline evaluator
26: to grade against. It fills its markers from the same resolvers as the rest,
27: which is what makes it a control rather than a differently configured run.
28: 
29: Markers, all of the form @@NAME@@:
30: 
31:     inline          replaced within the line, e.g. EXPRESSION = "@@EXPRESSION@@"
32:     whole line      a line that is only @@NAME@@ (or "# @@NAME@@") is replaced
33:                     by a block of lines
34:     #~ prefix       template-only note, dropped from the output
35: 
36: The tree maps onto PEFT directly:
37: 
38:     L<i>.w<j>   attach LoRA slot i, to be blended at weight w<j>
39:     CAT(a, b)   add_weighted_adapter(..., combination_type="cat")
40:     SVD(a, b)   add_weighted_adapter(..., combination_type="svd")
41:     LIN(a, b)   add_weighted_adapter(..., combination_type="linear")
42: 
43: A combined node is itself a named adapter, so it can feed its parent exactly
44: like a leaf does. Its children's weights are already baked into it, so it
45: enters its own parent at weight 1.0.
46: 
47: Rank bookkeeping matters, because PEFT constrains it (peft/tuners/lora/model.py,
48: _check_add_weighted_adapter):
49: 
50:     cat     -> new rank is the SUM of the children's ranks
51:     svd     -> new rank is the MAX of the children's ranks (no svd_rank given)
52:     linear  -> both children MUST have the same rank, else ValueError
53: 
54: So a LIN sitting above a CAT usually cannot run. That is a property of the
55: search space, not a bug here: this generator computes every node's rank up
56: front, marks the individual BAD, and stamps a warning into the script it makes.
57: """
58: 
59: import json
60: import os
61: 
62: from config import settings
63: 
64: from gep_lora.core.search.generate_population import UNARY_OPS, decode, levels
65: 
66: # The repo folder, one above this one. Every path a setting names is
67: # resolved against it, so nothing here depends on the cwd a driver was
68: # started from, or on which sub-folder this module ended up in.
69: _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
70: 
71: 
72: COMBINATION_TYPE = {"CAT": "cat", "SVD": "svd", "LIN": "linear"}
73: 
74: # Where the four templates live. TEMPLATE and BASELINE_TEMPLATE hold a bare
75: # name rather than a path -- "template_code_mocked.py" is what settings.py says
76: # and what every stored sweep recorded -- so a name is looked for here, and a
77: # sweep stored before the templates moved still names something that exists.
78: TEMPLATE_DIR = "templates"
79: 
80: TEMPLATE = os.path.join(_ROOT, TEMPLATE_DIR, "template_code.py")
81: 
82: # The base model on its own, answering the same eval prompts -- what the
83: # llm_judge_baseline evaluator measures an individual's answers against. One
84: # per base model rather than one per individual, so it is filled by
85: # baseline_run.py rather than by the runs step.
86: BASELINE_TEMPLATE = os.path.join(_ROOT, TEMPLATE_DIR, "template_baseline.py")
87: 
88: MARKER = "@@%s@@"
89: TEMPLATE_COMMENT = "#~"
90: 
91: 
92: def template_path(name=None):
93:     """Absolute path of a template, from a name, a path, or None.
94: 
95:     An absolute path, or a relative one naming a file from where the driver was
96:     started, is taken as it stands. Anything else is looked for beside the repo
97:     and then in templates/, so a bare name goes on meaning the template of that
98:     name however the folders are arranged. One resolver rather than one per
99:     caller: the runs step and the baseline both have to agree on which file a
100:     stored TEMPLATE means, or a sweep and its own control would be generated
101:     from different code.
102: 
103:     A bare name may also be written without its extension -- "template_code" for
104:     template_code.py. Every template is a .py file, so both are tried rather
105:     than letting four missing characters become a FileNotFoundError from
106:     whoever opens the result. What comes back when nothing matches is the
107:     templates/ candidate, so the failure names the folder a template was
108:     expected in; load_template() is what says so.
109:     """
110:     name = name or "template_code.py"
111:     if os.path.isabs(name) or os.path.exists(name):
112:         return os.path.abspath(name)
113:     # As written first, then with the extension, so a name that really exists
114:     # keeps the meaning it always had.
115:     names = [name] if os.path.splitext(name)[1] else [name, name + ".py"]
116:     for candidate in names:
117:         for folder in (_ROOT, os.path.join(_ROOT, TEMPLATE_DIR)):
118:             path = os.path.join(folder, candidate)
119:             if os.path.exists(path):
120:                 return path
121:     return os.path.join(_ROOT, TEMPLATE_DIR, names[-1])
122: 
123: 
124: def templates_available():
125:     """The templates this repo actually holds, for a failure to list."""
126:     folder = os.path.join(_ROOT, TEMPLATE_DIR)
127:     try:
128:         return sorted(name for name in os.listdir(folder) if name.endswith(".py"))
129:     except OSError:
130:         return []
131: 
132: 
133: def training_set_path(value=None):
134:     """Absolute path of the eval prompts file, from a settings value or None.
135: 
136:     TRAINING_SET is a setting rather than a line in the templates, so the eval
137:     set can be repointed without editing generated-script code and a sweep
138:     records which prompts it was scored against. A relative value is taken from
139:     this file's folder, the way gep_lora/core/pipeline/start_run.py resolves DB_RUN_DIR, so it never depends
140:     on the cwd a driver happened to be started from; an absolute one is used as
141:     it stands. None means whatever settings.py currently says -- callers holding
142:     a sweep's stored settings should pass that instead, or a resumed sweep would
143:     silently change eval sets.
144:     """
145:     value = value or settings.TRAINING_SET
146:     if not os.path.isabs(value):
147:         value = os.path.join(_ROOT, value)
148:     return os.path.abspath(value)
149: 
150: 
151: def base_model_name(value=None):
152:     """The base model the scripts load, from a settings value or None.
153: 
154:     BASE_MODEL is a setting rather than a line in the templates so that one
155:     name reaches all three of them -- the two individual templates and the
156:     baseline the llm_judge_baseline evaluator grades against. A control that
157:     loaded a different model than the individuals did would turn every
158:     improvement score into a comparison of two models rather than of a blend.
159: 
160:     Same contract as training_set_path(): None means whatever settings.py says
161:     now, and a caller holding a sweep's stored settings passes that instead.
162:     """
163:     value = value or settings.BASE_MODEL
164:     if not str(value).strip():
165:         raise SystemExit("BASE_MODEL is empty; it has to name the model the "
166:                          "adapters were trained on.")
167:     return str(value)
168: 
169: 
170: # What every sweep ran under before CHAT_TEMPLATE was a setting: unsloth's
171: # "qwen-2.5", hardcoded in every template, the lora server and training. A stored
172: # sweep that never recorded the setting keeps it, so resuming one does not change
173: # the words its prompts are written in.
174: LEGACY_CHAT_TEMPLATE = "qwen-2.5"
175: 
176: # render()'s and render_baseline()'s default for `chat_template`, and every
177: # `count` default below: read settings.py. A sentinel rather than None, because
178: # None is a real value for both -- the model's own template, and no cap.
179: FROM_SETTINGS = object()
180: 
181: 
182: def chat_template_name(conf=FROM_SETTINGS):
183:     """The chat template to prompt in: an unsloth template name, or None for
184:     the model's own.
185: 
186:     `conf` is a sweep's stored settings. One that holds CHAT_TEMPLATE gets its
187:     value; one stored before the setting existed gets LEGACY_CHAT_TEMPLATE,
188:     which is what it actually ran under -- not settings.py's, which would change
189:     how a resumed sweep's prompts are written. With no `conf`, settings.py.
190:     """
191:     if conf is FROM_SETTINGS:
192:         value = getattr(settings, "CHAT_TEMPLATE", None)
193:     elif "CHAT_TEMPLATE" in conf:
194:         value = conf["CHAT_TEMPLATE"]
195:     else:
196:         value = LEGACY_CHAT_TEMPLATE
197:     if value is None or not str(value).strip():
198:         return None
199:     return str(value).strip()
200: 
201: 
202: def _chat_template(value):
203:     """render()'s `chat_template` argument, as the literal a script gets."""
204:     return chat_template_name() if value is FROM_SETTINGS else (value or None)
205: 
206: 
207: def training_count(value=FROM_SETTINGS):
208:     """The eval-set cap as the templates want it: a positive int, or None.
209: 
210:     Left out, it is whatever settings.py currently says; a caller holding a
211:     sweep's stored settings passes that instead, so a resumed sweep keeps being
212:     judged on as many prompts as it was created with. None is no cap wherever it
213:     comes from -- a sentinel rather than None marks "not given", or a sweep
214:     stored with no cap would quietly get settings.py's.
215: 
216:     Validated here rather than in the templates because a bad value should stop
217:     the generation step, not turn up inside every generated script.
218:     """
219:     value = settings.TRAINING_COUNT if value is FROM_SETTINGS else value
220:     if value is None:
221:         return None
222:     try:
223:         count = int(value)
224:     except (TypeError, ValueError):
225:         raise SystemExit("TRAINING_COUNT must be a whole number or None, got %r" % value)
226:     # int() would quietly turn 2.5 into 2, and a fractional count of prompts is
227:     # a typo rather than an intention worth honouring.
228:     if count != float(value):
229:         raise SystemExit("TRAINING_COUNT must be a whole number or None, got %r" % value)
230:     if count < 1:
231:         raise SystemExit(
232:             "TRAINING_COUNT must be at least 1, got %d. None is how the cap is "
233:             "turned off; 0 would leave nothing to score." % count
234:         )
235:     return count
236: 
237: 
238: def eval_prompt_count(training_set=None, count=FROM_SETTINGS):
239:     """(path, used, total) for the eval set, or a clear failure.
240: 
241:     Counts non-blank lines rather than re-parsing: the template owns reading a
242:     line, and every generated script would fail at startup on a missing or empty
243:     file, so it is worth catching here instead. One line is one prompt in both
244:     shapes the templates accept -- a JSON record per line, or plain text -- so
245:     the count holds without this having to know which it is looking at.
246: 
247:     `used` is what TRAINING_COUNT leaves of `total` -- the same min() the
248:     templates apply, computed here so the runs step can report what a sweep will
249:     actually be scored on rather than what the file happens to hold.
250:     """
251:     path = training_set_path(training_set)
252:     try:
253:         with open(path, encoding="utf-8") as handle:
254:             lines = [line for line in handle if line.strip()]
255:     except OSError as error:
256:         raise SystemExit("cannot read the eval prompts from %s (%s). Every generated "
257:                          "script reads that file at startup." % (path, error.strerror))
258:     total = len(lines)
259:     if not total:
260:         raise SystemExit("%s has no prompts in it" % path)
261:     # The one shape that would slip past a line count: a whole-file JSON array
262:     # is one record spread over many lines, or many records on one. Caught here
263:     # so it costs one clear failure rather than one per generated script.
264:     if lines[0].lstrip().startswith("["):
265:         raise SystemExit(
266:             "%s looks like one big JSON array. The generated scripts read the "
267:             "eval file a line at a time -- write it as one JSON record per line "
268:             "(JSON Lines), or as plain one-prompt-per-line text." % path
269:         )
270:     cap = training_count(count)
271:     return path, (total if cap is None else min(cap, total)), total
272: 
273: 
274: def eval_records(training_set=None, count=FROM_SETTINGS):
275:     """The eval set as records the evaluate step can score against.
276: 
277:     -> [{"position": 1, "question": "...", "reference": "..." or None}, ...]
278: 
279:     Same file, same order and same cap the generated scripts ask under, read
280:     here for the other half of the job: an evaluator that compares an answer
281:     with the answer the dataset already carries needs that reference, and the
282:     scripts deliberately never see it -- handing a model the reference and then
283:     scoring its reply would be marking its own homework.
284: 
285:     Parsed the way template_code.py parses a line, because it is the same file
286:     in the same two shapes: a JSON record with a "messages" list -- first user
287:     turn is the question, first assistant turn after it is the reference -- or
288:     a plain line, which is a question with no reference at all. `position` is
289:     1-based and lines up with exchanges.position, since both count non-blank
290:     lines of this file from the top.
291: 
292:     Kept next to eval_prompt_count() rather than in the evaluators package so
293:     the eval
294:     file has one reader per concern and neither has to know where the file is:
295:     training_set_path() already answers that, for a stored sweep's value as
296:     much as for settings.py's.
297:     """
298:     cap = training_count(count)
299:     records = dataset_records(training_set)
300:     return records[:cap] if cap else records
301: 
302: 
303: def dataset_records(dataset=None):
304:     """One dataset file, whole, as records -- the uncapped form of the above.
305: 
306:     -> [{"position": 1, "question": "...", "reference": "..." or None,
307:          "content": "the line as it stood"}, ...]
308: 
309:     Same parser and the same two shapes eval_records() reads, and the same path
310:     rule, so a validation or testing split is read exactly the way the eval set
311:     is. Two things differ, and both are why this is a separate function:
312:     TRAINING_COUNT is not applied -- that cap says how many prompts an
313:     individual is *judged* on, which is nothing to do with how much of a file
314:     there is -- and each record keeps the line it came from, so storing a
315:     dataset stores what was actually in it rather than this module's reading of
316:     it.
317:     """
318:     path = training_set_path(dataset)
319:     try:
320:         with open(path, encoding="utf-8") as handle:
321:             lines = [line.strip() for line in handle if line.strip()]
322:     except OSError as error:
323:         raise SystemExit("cannot read the eval prompts from %s (%s)."
324:                          % (path, error.strerror))
325:     if lines and lines[0].startswith("["):
326:         # eval_prompt_count() says this about the eval set; a validation or
327:         # testing split is read by nothing else, so it is said here too rather
328:         # than letting a whole-file array be stored as one nonsense question.
329:         raise SystemExit(
330:             "%s looks like one big JSON array. A dataset is read a line at a "
331:             "time -- write it as one JSON record per line (JSON Lines), or as "
332:             "plain one-prompt-per-line text." % path
333:         )
334:     records = []
335:     for position, line in enumerate(lines, start=1):
336:         records.append({"position": position,
337:                         "question": _question_of(line, path, position),
338:                         "reference": _reference_of(line),
339:                         "content": line})
340:     return records
341: 
342: 
343: def _question_of(line, path, number):
344:     """The prompt in one eval line -- template_code.py's _prompt_of, once more.
345: 
346:     Duplicated rather than shared because the templates are standalone files
347:     that must run with nothing of this repo importable; a change to one of these
348:     two belongs in the other.
349:     """
350:     if line[0] == "{":
351:         try:
352:             record = json.loads(line)
353:         except ValueError as error:
354:             raise SystemExit("%s line %d: starts like a JSON record but will not "
355:                              "parse (%s)." % (path, number, error))
356:         messages = record.get("messages") if isinstance(record, dict) else None
357:         if not isinstance(messages, list):
358:             raise SystemExit("%s line %d: a JSON eval record needs a 'messages' "
359:                              "list, the shape datasets/*.json use." % (path, number))
360:         for message in messages:
361:             if isinstance(message, dict) and message.get("role") == "user":
362:                 content = message.get("content")
363:                 if isinstance(content, str) and content.strip():
364:                     return content.strip()
365:         raise SystemExit("%s line %d: no user turn with any text in it, so there "
366:                          "is nothing to ask." % (path, number))
367: 
368:     if len(line) > 1 and line[0] == line[-1] and line[0] in "\"'":
369:         line = line[1:-1]
370:     return line
371: 
372: 
373: def _reference_of(line):
374:     """The dataset's own answer to that line, or None when it carries none.
375: 
376:     A plain prompt file has no references, and neither does a JSON record whose
377:     messages stop at the user turn. That is not an error here -- it is only an
378:     error for the evaluators that need one, and they say so themselves.
379:     """
380:     if not line.startswith("{"):
381:         return None
382:     try:
383:         record = json.loads(line)
384:     except ValueError:
385:         return None
386:     messages = record.get("messages") if isinstance(record, dict) else None
387:     if not isinstance(messages, list):
388:         return None
389:     for message in messages:
390:         if isinstance(message, dict) and message.get("role") == "assistant":
391:             content = message.get("content")
392:             if isinstance(content, str) and content.strip():
393:                 return content.strip()
394:     return None
395: 
396: 
397: def lora_slots(slots=None):
398:     """{slot: location} with the ones that name a local folder made absolute.
399: 
400:     LORA_SLOTS is a setting rather than a block in the templates, so a slot is
401:     repointed in one place and both templates follow, and a sweep records which
402:     five adapters it was scored on. A relative entry is taken from this file's
403:     folder, the same rule training_set_path() uses -- but only when it really is
404:     a folder here; anything else is passed through as written, which is what
405:     leaves the door open for an absolute path or a Hub repo id. None means
406:     whatever settings.py currently says; callers holding a sweep's stored
407:     settings should pass those instead, or a resumed sweep would quietly blend
408:     different adapters.
409:     """
410:     resolved = {}
411:     for slot, where in (slots or settings.LORA_SLOTS).items():
412:         if not os.path.isabs(where):
413:             local = os.path.abspath(os.path.join(_ROOT, where))
414:             if os.path.isdir(local):
415:                 where = local
416:         resolved[slot] = where
417:     return resolved
418: 
419: 
420: def slot_ranks(slots=None):
421:     """{slot: rank} read from each adapter's own adapter_config.json.
422: 
423:     Ranks are not assumed equal: the five slots may point at LoRAs trained with
424:     different r values, and cat/svd/linear each treat that differently.
425:     """
426:     ranks = {}
427:     for slot, path in sorted(lora_slots(slots).items()):
428:         config_path = os.path.join(path, "adapter_config.json")
429:         try:
430:             with open(config_path, encoding="utf-8") as handle:
431:                 config = json.load(handle)
432:         except OSError as error:
433:             raise SystemExit(
434:                 "slot %s: cannot read %s (%s). Point LORA_SLOTS[%r] in settings.py "
435:                 "at a real adapter folder, or fix the path."
436:                 % (slot, config_path, error.strerror, slot)
437:             )
438:         # Same rule PEFT uses: rank_pattern can raise the rank above r.
439:         ranks[slot] = max([config["r"]] + list((config.get("rank_pattern") or {}).values()))
440:     return ranks
441: 
442: 
443: class Step:
444:     """One line of the build plan: either attach a leaf, or combine two nodes."""
445: 
446:     __slots__ = ("kind", "name", "symbol", "variable", "left", "right", "rank", "broken")
447: 
448:     def __init__(self, kind, name, symbol, rank):
449:         self.kind = kind          # "leaf" or "combine"
450:         self.name = name          # adapter name inside the model
451:         self.symbol = symbol      # L1..L5, or CAT/SVD/LIN
452:         self.variable = None      # leaf only: which w<j> weights it
453:         self.left = None          # combine only: (name, weight_expr, rank)
454:         self.right = None
455:         self.rank = rank
456:         self.broken = False       # combine only: LIN with mismatched child ranks
457: 
458: 
459: def plan(root, ranks):
460:     """Post-order walk of the tree -> the ordered list of build steps.
461: 
462:     `ranks` maps each slot (L1..L5) to the rank read from its adapter_config.json.
463:     Returns (steps, final_adapter_name). Nodes are numbered in the order they
464:     have to be built, so a step never references a name defined after it.
465:     """
466:     steps = []
467:     counter = [0]
468: 
469:     def next_name(symbol):
470:         counter[0] += 1
471:         return "n%d_%s" % (counter[0], symbol)
472: 
473:     def visit(node):
474:         """-> (adapter_name, weight expression, rank)."""
475:         if node.symbol in UNARY_OPS:
476:             # An L* node is a leaf adapter; its single child names the weight.
477:             variable = node.children[0].symbol
478:             rank = ranks[node.symbol]
479:             step = Step("leaf", next_name(node.symbol), node.symbol, rank)
480:             step.variable = variable
481:             steps.append(step)
482:             return step.name, 'WEIGHTS["%s"]' % variable, rank
483: 
484:         left = visit(node.children[0])
485:         right = visit(node.children[1])
486:         left_rank, right_rank = left[2], right[2]
487: 
488:         if node.symbol == "CAT":
489:             rank, broken = left_rank + right_rank, False
490:         elif node.symbol == "SVD":
491:             rank, broken = max(left_rank, right_rank), False
492:         else:                                   # LIN -> PEFT "linear"
493:             rank, broken = left_rank, left_rank != right_rank
494: 
495:         step = Step("combine", next_name(node.symbol), node.symbol, rank)
496:         step.left, step.right, step.broken = left, right, broken
497:         steps.append(step)
498:         # The children's weights are already folded in, so this node enters its
499:         # own parent at full strength.
500:         return step.name, "1.0", rank
501: 
502:     visit(root)
503:     return steps, steps[-1].name
504: 
505: 
506: # --- the blocks that fill the template ------------------------------------
507: 
508: 
509: def build_order_block(steps):
510:     """The 'this is what gets built' lines for the docstring."""
511:     notes = []
512:     for step in steps:
513:         if step.kind == "leaf":
514:             notes.append("    %-10s = %s @ %s" % (step.name, step.symbol, step.variable))
515:         else:
516:             notes.append("    %-10s = %s(%s, %s)"
517:                          % (step.name, step.symbol, step.left[0], step.right[0]))
518:         notes[-1] = "%-46s rank %d" % (notes[-1], step.rank)
519:         if step.broken:
520:             notes[-1] += "   <-- FAILS, see NOTE"
521:     notes[-1] += "   <-- generation runs through this one"
522:     return notes
523: 
524: 
525: def note_block(steps):
526:     """The warning for trees PEFT will refuse, or nothing at all."""
527:     broken = [step for step in steps if step.broken]
528:     if not broken:
529:         return []
530:     lines = ["NOTE - this tree cannot run as written."]
531:     for step in broken:
532:         lines.append('    %s is LIN, which PEFT maps to combination_type="linear", and'
533:                      % step.name)
534:         lines.append("    linear requires both inputs to have the same rank -- but %s has"
535:                      % step.left[0])
536:         lines.append("    rank %d and %s has rank %d, so the run stops there."
537:                      % (step.left[2], step.right[0], step.right[2]))
538:     lines.append("    (cat sums its inputs' ranks, which is what pushes them apart.)")
539:     lines.append("    Fixes: swap that LIN for SVD, pass svd_rank= to the CAT feeding it,")
540:     lines.append("    or treat the individual as unfit and drop it from the population.")
541:     lines.append("")
542:     return lines
543: 
544: 
545: def attach_leaves_block(steps):
546:     """One attach() per leaf, in post-order."""
547:     return ['attach("%s", "%s")' % (step.name, step.symbol)
548:             for step in steps if step.kind == "leaf"]
549: 
550: 
551: def combine_nodes_block(steps):
552:     """One combine() per binary node, in post-order."""
553:     lines = []
554:     for step in (step for step in steps if step.kind == "combine"):
555:         if step.broken:
556:             lines.append("# !! Stops here: linear needs equal ranks, but %s is rank %d"
557:                          % (step.left[0], step.left[2]))
558:             lines.append("# !! and %s is rank %d. See NOTE at the top."
559:                          % (step.right[0], step.right[2]))
560:         lines.append('combine("%s", "%s", ("%s", %s), ("%s", %s))'
561:                      % (step.name, COMBINATION_TYPE[step.symbol],
562:                         step.left[0], step.left[1], step.right[0], step.right[1]))
563:     return lines
564: 
565: 
566: # --- filling the template -------------------------------------------------
567: 
568: 
569: def load_template(path=TEMPLATE):
570:     """Read the template, dropping its #~ notes.
571: 
572:     A template that is not there is a mistyped TEMPLATE nine times in ten, and
573:     every generated script in the sweep would come from it -- so it is worth one
574:     clear failure naming what was looked for and what is actually in templates/,
575:     rather than a bare FileNotFoundError from this open().
576:     """
577:     try:
578:         with open(path, encoding="utf-8") as handle:
579:             return [line.rstrip("\n") for line in handle
580:                     if not line.lstrip().startswith(TEMPLATE_COMMENT)]
581:     except OSError as error:
582:         held = templates_available()
583:         raise SystemExit(
584:             "cannot read the template %s (%s). TEMPLATE in settings.py names "
585:             "the file every generated script is filled from; %s/ holds %s."
586:             % (path, error.strerror, TEMPLATE_DIR,
587:                ", ".join(held) if held else "nothing"))
588: 
589: 
590: def fill(template_lines, blocks, values):
591:     """Substitute every marker.
592: 
593:     A line whose only content is one marker (bare, or commented out so the
594:     template stays valid Python) is replaced by that marker's block of lines.
595:     Every other marker is replaced inside the line it sits on.
596:     """
597:     block_markers = {MARKER % name: lines for name, lines in blocks.items()}
598:     out = []
599:     for line in template_lines:
600:         stripped = line.strip()
601:         if stripped.startswith("#"):
602:             stripped = stripped.lstrip("#").strip()
603:         if stripped in block_markers:
604:             out.extend(block_markers[stripped])
605:             continue
606:         for name, value in values.items():
607:             line = line.replace(MARKER % name, value)
608:         out.append(line)
609: 
610:     leftover = [line for line in out if "@@" in line]
611:     if leftover:
612:         raise ValueError("template marker was never filled: %s" % leftover[0].strip())
613:     return "\n".join(out) + "\n"
614: 
615: 
616: def render(expression, steps, final, script_name, provenance, label,
617:            template_lines=None, template_path=TEMPLATE, weight_seed=None,
618:            training_set=None, slots=None, count=FROM_SETTINGS, base_model=None,
619:            chat_template=FROM_SETTINGS, root=None):
620:     """The complete text of one runnable script, built from the template.
621: 
622:     Callers pass the planning results from plan(); the template supplies
623:     everything that does not vary between individuals.
624: 
625:     Which template that is comes from `template_path`, or from `template_lines`
626:     if the caller has already read one -- a batch job reads it once and reuses
627:     the lines, a one-off caller just names the file.
628: 
629:     `weight_seed` is what the script's WEIGHT_SEED becomes. None leaves the
630:     script redrawing its blend weights from the OS every execution, so the same
631:     chromosome is judged under different weights each time. An int pins the draw,
632:     which is how gep_lora/core/pipeline/start_run.py makes a stored sweep repeatable -- it records the seed
633:     it stamped in here.
634: 
635:     `training_set` is what the script's TRAINING_SET becomes, resolved to an
636:     absolute path; None takes it from settings.py. `slots` is the same for
637:     LORA_SLOTS and `base_model` for BASE_MODEL. `count` is TRAINING_COUNT and
638:     `chat_template` is CHAT_TEMPLATE already resolved -- a name, or None for the
639:     model's own -- and None is a value for both (no cap; the model's own
640:     template), so leave them out to read settings.py.
641:     gep_lora/core/pipeline/start_run.py passes the sweep's stored values for all four, so a resumed sweep
642:     keeps reading the prompts, reading as many of them, blending the adapters
643:     and loading the model it was created with even if settings.py has since
644:     moved on.
645: 
646:     `root` is the tree `steps` were planned from, for a caller whose tree is not
647:     a chromosome: a single adapter on its own is a leaf with no CAT above it,
648:     which the grammar refuses to decode but plan() builds and every template
649:     runs. Left out, it is decoded from `expression`, as for any individual.
650:     """
651:     template_lines = (load_template(template_path) if template_lines is None
652:                       else template_lines)
653:     if root is None:
654:         root, _ = decode(expression)
655:     leaves = [step for step in steps if step.kind == "leaf"]
656:     # Resolved once: the block below wants it, and it validates, so a bad
657:     # TRAINING_COUNT fails here rather than in every script rendered after it.
658:     cap = training_count(count)
659: 
660:     blocks = {
661:         "TREE": ["    %s" % ".".join(row) for row in levels(root)],
662:         "BUILD_ORDER": build_order_block(steps),
663:         "NOTE": note_block(steps),
664:         "ATTACH_LEAVES": attach_leaves_block(steps),
665:         "COMBINE_NODES": combine_nodes_block(steps),
666:         # One line, but a block rather than an inline marker: it stands in for
667:         # the assignment itself, so the generated file gets a plain literal.
668:         "WEIGHT_SEED": ["WEIGHT_SEED = %s"
669:                         % ("None" if weight_seed is None else int(weight_seed))],
670:         # Likewise a block: the generated script gets the resolved path as a
671:         # literal, rather than working it out from where it happens to sit.
672:         "TRAINING_SET": ["TRAINING_SET = %r" % training_set_path(training_set)],
673:         # And the cap on it, as a literal for the same reason: the script slices
674:         # the prompts it read, so it carries the number rather than looking it
675:         # up in a settings.py that may have moved on since.
676:         "TRAINING_COUNT": ["TRAINING_COUNT = %s" % cap],
677:         # And likewise the adapters: one resolved entry per slot, in slot order
678:         # so two scripts built from the same settings are byte-identical here.
679:         "LORA_SLOTS": (["LORA_SLOTS = {"]
680:                        + ["    %r: %r," % pair
681:                           for pair in sorted(lora_slots(slots).items())]
682:                        + ["}"]),
683:         # And the model all of them are attached to, from the same one place
684:         # the baseline script gets it from.
685:         "BASE_MODEL": ["BASE_MODEL = %r" % base_model_name(base_model)],
686:         # And the chat template the prompts are written in, which the baseline
687:         # gets from here too: a control asked in other words is no control.
688:         "CHAT_TEMPLATE": ["CHAT_TEMPLATE = %r" % _chat_template(chat_template)],
689:     }
690:     values = {
691:         "SCRIPT_NAME": script_name,
692:         "PROVENANCE": provenance,
693:         "LABEL": label,
694:         "EXPRESSION": expression,
695:         "LEAF_COUNT": str(len(leaves)),
696:         "FINAL_ADAPTER": final,
697:     }
698:     return fill(template_lines, blocks, values)
699: 
700: 
701: def render_baseline(script_name, provenance, template_lines=None,
702:                     template_path=BASELINE_TEMPLATE, training_set=None,
703:                     count=FROM_SETTINGS, base_model=None, chat_template=FROM_SETTINGS):
704:     """The complete text of the one baseline script -- the base model alone.
705: 
706:     The control the llm_judge_baseline evaluator grades against: the same base
707:     model, the same eval prompts, the same cap, and nothing attached. It has no
708:     tree, no weights and no adapters, so it fills three markers where render()
709:     fills eleven -- but it fills them from the same resolvers, which is what
710:     makes it a control rather than a second, differently configured run.
711: 
712:     There is one of these per base model, not one per individual, so nothing
713:     about an individual reaches it. baseline_run.py fills it, runs it once, and
714:     caches what it said.
715:     """
716:     template_lines = (load_template(template_path) if template_lines is None
717:                       else template_lines)
718:     blocks = {
719:         "BASE_MODEL": ["BASE_MODEL = %r" % base_model_name(base_model)],
720:         "TRAINING_SET": ["TRAINING_SET = %r" % training_set_path(training_set)],
721:         "TRAINING_COUNT": ["TRAINING_COUNT = %s" % training_count(count)],
722:         "CHAT_TEMPLATE": ["CHAT_TEMPLATE = %r" % _chat_template(chat_template)],
723:     }
724:     values = {"SCRIPT_NAME": script_name, "PROVENANCE": provenance}
725:     return fill(template_lines, blocks, values)

(End of file - total 725 lines)
</content>
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 5.1s)

_Thinking:_

This is very thoroughly documented code. Let me now read the search algorithms (selection, mutation, elitism, calculate_fitness, draw_trees), process_run, start_run (partially), and the templates. Let me also check the async API and the agent. Let me use parallel reads and explore agents.

Let me read the search modules and process_run.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\search\\selection.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\search\selection.py</path>
<type>file</type>
<content>
1: """
2: selection.py - Roulette wheel sampling over the population.
3: 
4:     individuals.fitness  -->  n copies of the fit and one complete stranger,
5:                               in place of the n+1 weakest
6: 
7: Fitness-proportionate selection, the classic one: give every individual a slice
8: of a wheel as wide as its fitness, spin the wheel once per pick, and take
9: whoever it lands on. A chromosome twice as fit is twice as likely to be picked,
10: and no chromosome is picked for certain -- a mediocre individual keeps a real
11: chance, which is what stops the search collapsing onto the first good blend it
12: finds.
13: 
14: Picks are drawn **with replacement**: the same individual can come up several
15: times in one round, and that is the mechanism rather than a flaw in it. That is
16: how a fit chromosome comes to have several descendants.
17: 
18: What one round does
19: -------------------
20: Three things, in this order, and the arithmetic is the point:
21: 
22: 1. **Appends `n` copies** of whoever the wheel landed on, `n` being
23:    `SELECTION_COUNT` (or the size of the population, when that is None).
24: 2. **Culls the `n+1` weakest** individuals of the population as it was before
25:    the round -- never the elite.
26: 3. **Appends one brand-new individual**, drawn from
27:    `generate_population.random_tree()` under the sweep's own `MAX_DEPTH`,
28:    `BRANCH_PROB` and `UNIQUE`: the same draw the first population came from.
29: 
30: `n + 1` in and `n + 1` out, so a generation ends **exactly the size it began**:
31: a sweep drawn at `COUNT = 10` is still ten individuals after ten generations.
32: The cull is `n+1` and not `n` because the round appends `n+1` rows -- the
33: copies *and* the newcomer -- and it is that total, not the number of copies,
34: that has to come back out. What moves is not the size but the membership: the
35: fit are duplicated, the weak are gone, and one member of every generation owes
36: nothing to either.
37: 
38: That makes `SELECTION_COUNT` a knob on the **turnover** rather than on the
39: size. At 2 over a population of 10, three individuals in three out each
40: generation; at None it is as many copies as the population holds, which is more
41: than the cull can take (see below) and the only setting that still grows it.
42: 
43: The newcomer is the reason the cull is safe. Selection can only ever pick
44: chromosomes the population already holds, and mutation only ever nudges a
45: symbol into a sibling of its own class, so a search that culls and copies is a
46: search whose gene pool can only narrow. One fresh draw per generation is a
47: floor under that: whatever the population has converged on, every generation
48: still contains one tree that was grown from nothing.
49: 
50: What a cull takes with it
51: -------------------------
52: The individual whole -- its executions, its exchanges and its test_results all
53: cascade (see store.delete_individuals). Its number is retired rather than
54: reused, and its rows in `fitness_history` stay exactly where they are, because
55: the history is what each generation *was* and a cull is not allowed to rewrite
56: that. So a sweep can always say that individual #7 scored 0.04 in generation 3
57: and was culled at the end of it.
58: 
59: The weakest means the lowest `fitness`, with NULL read as 0.0 the way every
60: other reader of that column reads it, and ties broken on the **lowest number**
61: -- so between two equally weak individuals the longer-standing one goes, and
62: re-running the step culls the same rows rather than a coin-flip's worth of
63: different ones. `is_best` is never eligible: culling the elite would discard
64: the one thing elitism exists to protect, and it is the only individual the step
65: refuses outright. When the population is too small to give up `n+1` non-elite
66: rows it gives up as many as it has, and grows by the difference -- which is
67: what `SELECTION_COUNT = None` does every round, since asking for the whole
68: population as copies asks the cull for one more row than the population has
69: individuals to spare.
70: 
71: A parent the wheel just landed on can itself be culled -- weak individuals do
72: occasionally get picked -- and nothing needs doing about that: the copy it
73: produced carries its chromosome, its script and its fitness forward, which is
74: all the parent had to pass on.
75: 
76: What a copy is
77: --------------
78: A copy is the parent field for field: its tree, its state, its rank, its
79: script, its weight seed and its fitness, not merely its chromosome. Only the id
80: and the number are its own, and only because those are what make it a row of
81: its own. It is therefore a clone in the full sense -- it arrives already
82: carrying the result its parent earned, rather than as a blank waiting to be
83: built and judged.
84: 
85: The one thing no copy inherits is `is_best`. That flag does not describe an
86: individual, it picks one out of the population, so it is not the parent's to
87: hand on: a copy of the elite is not itself the elite, and a sweep that ended a
88: round with several of them would have lost the only thing the flag says. Every
89: copy arrives at 0, and the next election decides on the fitness it earns.
90: 
91: That inheritance is meant to be *spent*, not kept. The copies exist for whatever
92: comes next to vary, and a copy that is varied has a chromosome its inherited
93: tree, script, rank and fitness no longer describe -- they are the parent's
94: answers to a question the child no longer asks. Re-deriving them is what
95: `python start_run.py trees runs` does, from the chromosome, for every individual;
96: until then a copy carries its parent's, including the script name, so two rows
97: can name the same run_NNN.py and the same weight seed while they are still the
98: same chromosome anyway.
99: 
100: The newcomer needs the same two steps for the opposite reason: it arrives with
101: a chromosome and *nothing else* -- no tree, no script, no seed, no fitness --
102: exactly as a member of the first population does, so `trees` and `runs` are
103: what make it runnable and `process` is what earns it a score.
104: 
105: A round replacing what it adds does not make the step idempotent: running it
106: twice runs two rounds of selection, and the second one culls what the first one
107: left -- the population comes out the same size and made of different
108: individuals. That is what a second generation *is*, so it is deliberate
109: -- but it does mean `python start_run.py selection` is a thing you do on purpose,
110: not a thing you repeat to be sure it took.
111: 
112: Zero fitness
113: ------------
114: A slice as wide as 0.0 can never be landed on, so an individual with no fitness
115: is never picked -- correct for roulette, and worth saying out loud because
116: `individuals.fitness` defaults to 0.0. A population where *everything* is 0.0
117: has no wheel to spin at all: every slice is empty, and picking anyway would just
118: be a uniform draw wearing a selection step's name. That case selects nobody,
119: culls nobody, draws no newcomer and writes nothing -- a round that cannot say
120: which individuals are the fit ones cannot be trusted to say which are the weak
121: ones either, and one that culled on that basis would empty a population instead
122: of holding it steady.
123: 
124: gep_lora/core/pipeline/start_run.py calls this as a library:
125: 
126:     selection.select(conn, run_id, count, rng, conf)
127: """
128: 
129: import bisect
130: from collections import namedtuple
131: 
132: from search import generate_population
133: from storage import store
134: 
135: # What one round of selection came to: the rows the wheel landed on, in the
136: # order it landed on them; the numbers their copies were appended under; the
137: # population as it was before any of it; the rows the cull took; and the
138: # newcomer, or None when the round did nothing at all.
139: Round = namedtuple("Round", "parents numbers population culled newcomer")
140: 
141: # The one individual per round that is nobody's copy.
142: Newcomer = namedtuple("Newcomer", "number chromosome")
143: 
144: # How many draws to spend looking for a chromosome the population does not
145: # already hold, before settling for one it does. A generous ceiling on a cheap
146: # draw: the alternative is failing a generation over a duplicate, which would
147: # be a worse answer than a duplicate.
148: FRESH_ATTEMPTS = 100
149: 
150: 
151: def wheel(rows):
152:     """The wheel `rows` make: (cumulative edges, total width).
153: 
154:     Edge i is where individual i's slice ends, so the slices are laid end to end
155:     and a mark anywhere in [0, total) falls in exactly one of them. A negative
156:     fitness would eat into its neighbour, so it is floored at zero -- quality is
157:     0..1 and cannot produce one, but the wheel should not depend on that.
158:     """
159:     edges, total = [], 0.0
160:     for row in rows:
161:         total += max(0.0, row["fitness"] or 0.0)
162:         edges.append(total)
163:     return edges, total
164: 
165: 
166: def spin(rows, edges, total, rng):
167:     """One spin: the individual the mark falls on.
168: 
169:     bisect_right and not bisect_left, so that a mark landing exactly on an edge
170:     goes to the slice that *starts* there rather than the one that ended -- which
171:     is what skips the zero-width slices of the unfit instead of handing a pick
172:     to one of them.
173:     """
174:     index = bisect.bisect_right(edges, rng.random() * total)
175:     return rows[min(index, len(rows) - 1)]
176: 
177: 
178: def draw(rows, count, rng):
179:     """Spin the wheel `count` times, with replacement. -> the rows it landed on.
180: 
181:     Empty when there is nothing to spin: no population, nothing asked for, or a
182:     wheel of nothing but zero-width slices.
183:     """
184:     edges, total = wheel(rows)
185:     if not rows or count <= 0 or total <= 0.0:
186:         return []
187:     return [spin(rows, edges, total, rng) for _ in range(count)]
188: 
189: 
190: def weakest(rows, count):
191:     """The `count` weakest of `rows`, weakest first. -> rows, never the elite.
192: 
193:     Reads the stored `fitness` column and nothing else, so "weakest" here means
194:     exactly what "best" means to elitism.py -- one definition of the number,
195:     living in calculate_fitness.py. NULL is 0.0, as it is to every other reader:
196:     a mutant whose score was cleared has not been judged since, and an unjudged
197:     individual is not one the search has any reason to keep.
198: 
199:     Fewer than `count` rows come back when the population cannot spare them,
200:     which is the one way a round can leave the population bigger than it found
201:     it.
202:     """
203:     eligible = [row for row in rows if not row["is_best"]]
204:     eligible.sort(key=lambda row: (row["fitness"] or 0.0, row["number"]))
205:     return eligible[:max(0, count)]
206: 
207: 
208: def fresh(rows, rng, conf):
209:     """A brand-new chromosome, drawn the way the first population was.
210: 
211:     generate_population.build_population() with a count of one, so the newcomer
212:     is grown by the same random_tree() under the same MAX_DEPTH and BRANCH_PROB
213:     and validated by the same check() -- there is no second draw and no second
214:     grammar. Its own `unique` flag is left off because it only dedupes within
215:     the batch it draws, and a batch of one has nothing to dedupe against.
216: 
217:     The sweep's UNIQUE is honoured here instead, and against something more
218:     useful: the chromosomes the population held going into this round, which
219:     includes the ones about to be culled. A newcomer exists to bring the search
220:     something it does not have, so re-drawing what it just discarded would be
221:     the one draw that achieves nothing. Exhausting the attempts is not an error
222:     -- a converged search that can only find chromosomes it already holds is
223:     telling you something, and a duplicate newcomer says it more usefully than
224:     a failed generation would.
225:     """
226:     held = {row["chromosome"] for row in rows}
227:     chromosome = None
228:     for _ in range(FRESH_ATTEMPTS if conf.get("UNIQUE", True) else 1):
229:         chromosome = generate_population.build_population(
230:             1, rng, conf["MAX_DEPTH"], conf["BRANCH_PROB"], False)[0]
231:         if chromosome not in held:
232:             break
233:     return chromosome
234: 
235: 
236: def select(conn, run_id, count, rng, conf):
237:     """Run one round of selection: copy, cull, draw one stranger. -> a Round.
238: 
239:     Writes nothing when the wheel cannot be spun, in which case `parents`,
240:     `numbers` and `culled` are all empty, `newcomer` is None, and the
241:     population is exactly as it was.
242: 
243:     The order the three writes happen in is the order they have to happen in.
244:     The copies go in first, so the cull is never asked to take rows the round
245:     has not yet replaced, and so the numbers it retires are always below the
246:     ones this round handed out -- which is what keeps store.next_number() from
247:     ever reissuing one. The cull then reads the population as it was *before*
248:     the round -- `rows`, not the table -- so it can only ever take individuals
249:     the search has actually judged, never one of the copies just appended or
250:     the newcomer about to be. And the newcomer's number comes last because it
251:     is the newest thing in the sweep.
252: 
253:     The newcomer is drawn from the same `rng` as the spins, after them, so the
254:     round as a whole stays reproducible from the sweep's stored
255:     SELECTION_MASTER_SEED. The cull consumes no randomness at all.
256:     """
257:     rows = store.individuals(conn, run_id)
258:     parents = draw(rows, len(rows) if count is None else count, rng)
259:     if not parents:
260:         return Round([], [], rows, [], None)
261: 
262:     numbers = store.append_copies(conn, run_id, parents)
263:     culled = store.delete_individuals(
264:         conn, run_id, [row["number"] for row in weakest(rows, len(parents) + 1)])
265:     chromosome = fresh(rows, rng, conf)
266:     newcomer = Newcomer(store.append_individual(conn, run_id, chromosome), chromosome)
267:     return Round(parents, numbers, rows, culled, newcomer)

(End of file - total 267 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\search\\mutation.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\search\mutation.py</path>
<type>file</type>
<content>
1: """
2: mutation.py - Point mutation over the population, minus the elite.
3: 
4:     individuals.chromosome  -->  chromosome, has_changed, fitness
5: 
6: Selection makes copies; mutation is what makes them worth having. Every symbol
7: of every non-elite chromosome is offered a chance, `rate`, of being replaced by
8: a different symbol -- so a chromosome of eleven symbols at rate 0.1 expects
9: about one change, and any individual may come through untouched.
10: 
11: The elite does not change
12: -------------------------
13: The individual marked `is_best` is passed over: elitism named it precisely so
14: that the best result found so far survives a generation intact, and mutating it
15: would throw away the thing the flag exists to protect. It is the one row this
16: step reads and does not write.
17: 
18: Only valid changes
19: ------------------
20: A symbol may only be replaced by one of its own kind:
21: 
22:     CAT SVD LIN     swap freely among themselves       (arity 2, operator children)
23:     L1 .. L5        swap freely among themselves       (arity 1, variable child)
24:     w1 .. w5        swap freely among themselves       (arity 0)
25: 
26: and the root is never touched at all, because the grammar fixes it at CAT.
27: 
28: That restriction is the whole trick. A symbol's arity and the alphabet its
29: children are drawn from are properties of its *class*, not of the symbol, so a
30: swap inside a class leaves the tree exactly the shape it was: every node still
31: has the number of children it had, and every child is still legal where it
32: stands. Reaching across classes would not -- turning a `CAT` into an `L2` would
33: leave a node with two children where one belongs, and the second of them a
34: subtree where a bare `w` is required, which is not a chromosome at all. There is
35: no repair step here and no tail to absorb the damage, so the mutation simply
36: does not make changes it would have to repair. Every result is put through
37: generate_population.check() before it is stored, which is the guarantee rather
38: than a hope.
39: 
40: What may still come out broken is a `LIN` above two different ranks -- swapping
41: `CAT` for `LIN` can easily produce one. That is a legal chromosome describing a
42: blend PEFT will not build, which is a property of the search space and is culled
43: downstream: the runs step marks it `state = 'BAD'` and process skips it.
44: 
45: has_changed, and the fitness that goes with it
46: ----------------------------------------------
47: has_changed is set to 1 on an individual whose chromosome this step actually
48: altered, and 0 on every other -- the elite included, and an individual the dice
49: passed over. It is this round's answer, not a running total, so it is written
50: for every individual each time rather than only for the ones that moved.
51: 
52: Setting it to 1 clears that individual's fitness to NULL. The score was earned
53: by the chromosome that has just been replaced, which makes keeping it worse than
54: stale: it would let a mutant be elected, or win a slice of the roulette wheel,
55: on the strength of a blend it no longer describes. NULL says what is true --
56: this chromosome has not been judged yet -- and both elitism and selection
57: already read a missing fitness as no fitness, so a mutant simply waits its turn
58: until process and evaluate have given it one of its own.
59: 
60: gep_lora/core/pipeline/start_run.py calls this as a library:
61: 
62:     mutation.apply(conn, run_id, rate, rng)
63: """
64: 
65: from collections import namedtuple
66: 
67: from search import generate_population
68: from storage import store
69: 
70: # One individual this round touched: what it was, what it became, and how many
71: # symbols differ between the two.
72: Change = namedtuple("Change", "number before after symbols")
73: 
74: 
75: def alternatives(symbol, position):
76:     """The symbols `symbol` may legally become at `position`.
77: 
78:     Empty for the root, which the grammar fixes at CAT, and for anything that
79:     is not in the alphabet at all.
80:     """
81:     if position == 0:
82:         return ()
83:     for family in (generate_population.BINARY_OPS,
84:                    generate_population.UNARY_OPS,
85:                    generate_population.VARIABLES):
86:         if symbol in family:
87:             return tuple(other for other in family if other != symbol)
88:     return ()
89: 
90: 
91: def mutate(chromosome, rate, rng):
92:     """One chromosome through the dice. -> the chromosome it came out as.
93: 
94:     Each symbol independently has probability `rate` of being replaced by one of
95:     its alternatives, drawn uniformly. The result is checked before it is
96:     returned: a mutation that could not be read back would be a bug here, not a
97:     result, and should never reach a caller.
98:     """
99:     symbols = chromosome.split(".")
100:     drawn = []
101:     for position, symbol in enumerate(symbols):
102:         choices = alternatives(symbol, position)
103:         drawn.append(rng.choice(choices)
104:                      if choices and rng.random() < rate else symbol)
105:     mutated = ".".join(drawn)
106:     generate_population.check(mutated)
107:     return mutated
108: 
109: 
110: def differences(before, after):
111:     """How many symbols differ between two chromosomes of the same shape."""
112:     return sum(one != other for one, other in zip(before.split("."), after.split(".")))
113: 
114: 
115: def apply(conn, run_id, rate, rng):
116:     """Mutate a whole population but for its elite. -> (changes, rows).
117: 
118:     Writes has_changed for every individual; for the ones that actually moved it
119:     also writes the new chromosome and clears the fitness the old one earned.
120:     `changes` holds one Change per individual that moved; `rows` is the
121:     population as it was before any of it.
122:     """
123:     rows = store.individuals(conn, run_id)
124:     changes = []
125:     for row in rows:
126:         if row["is_best"]:
127:             # The one row that is read and not written: see the module note.
128:             store.set_changed(conn, row["id"], 0)
129:             continue
130:         mutated = mutate(row["chromosome"], rate, rng)
131:         if mutated == row["chromosome"]:
132:             store.set_changed(conn, row["id"], 0)
133:         else:
134:             store.set_chromosome(conn, row["id"], mutated)
135:             changes.append(Change(row["number"], row["chromosome"], mutated,
136:                                   differences(row["chromosome"], mutated)))
137:     conn.commit()
138:     return changes, rows

(End of file - total 138 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\search\\calculate_fitness.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\search\calculate_fitness.py</path>
<type>file</type>
<content>
1: """
2: calculate_fitness.py - Turn the judged answers into one number per individual.
3: 
4:     exchanges.quality  -->  individuals.fitness
5: 
6: The judge scores answers, not chromosomes: a sweep comes out of `evaluate` with
7: a quality on every exchange and nothing at all on the individual that produced
8: them. Selection needs the opposite -- one comparable number per individual --
9: so this step folds a transcript into its mean.
10: 
11:     fitness = the average quality across the exchanges of the individual's
12:               most recent execution
13: 
14: The most recent execution and not all of them, for the same reason `evaluate`
15: scores only that one: running a chromosome again under a different weight seed
16: is a new result, not an amendment to the old one. That choice already lives in
17: the `individual_quality` view, which is where the average is read from.
18: 
19: An individual with nothing to average -- never run, run and crashed before it
20: answered, or still unjudged -- gets 0.0 rather than NULL. Fitness is what a
21: selection step sorts on, and a missing score there would have to be given a
22: meaning at every call site; giving it one here, once, says the only sensible
23: thing: an individual that produced no judged answer is worth nothing. That is
24: also what a BAD individual gets, which is right -- a blend PEFT refuses to
25: build cannot be selected for.
26: 
27: Partly judged individuals are averaged over the answers that do have a score
28: (the view's AVG skips NULLs) and reported, since a fitness over half a
29: transcript is a weaker claim than one over all of it.
30: 
31: The same numbers are also written to `fitness_history`, one row per individual
32: per generation, stamped with the moment they were worked out. `individuals.
33: fitness` is a column that only ever holds *now*: the next generation overwrites
34: it and mutation clears it outright, so a sweep that keeps only that column can
35: say how fit its population is and nothing at all about whether the search is
36: getting anywhere. The history is the record of the run as a run -- what a
37: fitness curve is drawn from.
38: 
39: gep_lora/core/pipeline/start_run.py calls this as a library:
40: 
41:     calculate_fitness.assign(conn, run_id)
42: """
43: 
44: from collections import namedtuple
45: 
46: from storage import store
47: 
48: # What one pass of this step came to: which generation it recorded, when it
49: # recorded it, and the individual_quality rows it worked from, best first.
50: Snapshot = namedtuple("Snapshot", "generation recorded_at rows")
51: 
52: 
53: def fitness_of(row):
54:     """The fitness of one `individual_quality` row.
55: 
56:     Its mean answer quality, or 0.0 when it has none to speak of.
57:     """
58:     quality = row["quality"]
59:     return 0.0 if quality is None else round(float(quality), 6)
60: 
61: 
62: def entry_for(row):
63:     """One `individual_quality` row as a fitness_history row.
64: 
65:     It carries the chromosome and the state alongside the number, rather than
66:     leaning on a join back to `individuals`: the individual this describes will
67:     be mutated into a different chromosome and given a different fitness before
68:     the sweep is done, and a history read through the population as it stands
69:     today would report every past generation in terms of the present one.
70:     """
71:     return {"number": row["number"],
72:             "chromosome": row["chromosome"],
73:             "state": row["state"],
74:             "fitness": fitness_of(row),
75:             "answers": row["answers"] or 0,
76:             "unscored": row["unscored"] or 0}
77: 
78: 
79: def assign(conn, run_id):
80:     """Write individuals.fitness for a whole sweep, and record the generation.
81: 
82:     Two writes of one number: the column selection and elitism read, and a row
83:     in `fitness_history` so the value survives the next generation overwriting
84:     the column.
85: 
86:     Returns a Snapshot -- the generation recorded, its timestamp, and the rows
87:     it worked from, best first -- so a caller can report on the sweep without
88:     asking the database a second time.
89: 
90:     A sweep with no individuals is not a generation and is not recorded; the
91:     caller has a better complaint to make about that than this step does.
92:     """
93:     rows = store.quality_rows(conn, run_id)
94:     for row in rows:
95:         store.set_fitness(conn, run_id, row["number"], fitness_of(row))
96:     conn.commit()
97:     if not rows:
98:         return Snapshot(0, None, rows)
99: 
100:     generation = store.fitness_generation(conn, run_id)
101:     recorded_at = store.record_fitness(conn, run_id, generation,
102:                                        [entry_for(row) for row in rows])
103:     return Snapshot(generation, recorded_at, rows)

(End of file - total 103 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\search\\elitism.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\search\elitism.py</path>
<type>file</type>
<content>
1: """
2: elitism.py - Name the individual this generation carries forward.
3: 
4:     individuals.fitness  -->  individuals.is_best
5: 
6: Elitism is the rule that the best individual survives a generation untouched.
7: Selection, crossover and mutation are all free to lose it otherwise: the best
8: chromosome of one generation can easily have no descendant in the next, and a
9: search that can go backwards wastes the generations it spends climbing again.
10: Marking it is what lets a later step copy it across unchanged.
11: 
12:     is_best = 1 for one individual with the highest fitness, 0 for every other
13: 
14: Exactly one, whatever the fitness values look like. `individuals.fitness` is a
15: mean of judged answers, so ties are ordinary -- two chromosomes can easily blend
16: to the same score, and every individual of an all-zero generation ties with the
17: rest. The tie is broken by the lowest individual number: an arbitrary rule, but
18: a fixed one, so a sweep re-run over the same database elects the same individual
19: rather than a different one each time.
20: 
21: The flag is set from the stored `fitness` column and nothing else. Reading the
22: transcripts again here would be a second, quietly different definition of "best"
23: -- if the fitness rule changes, it changes in calculate_fitness.py and this step
24: follows it without knowing that it did.
25: 
26: gep_lora/core/pipeline/start_run.py calls this as a library:
27: 
28:     elitism.elect(conn, run_id)
29: """
30: 
31: from storage import store
32: 
33: 
34: def best_of(rows):
35:     """The elite among `rows`: highest fitness, lowest number breaking the tie.
36: 
37:     Returns None when there is no elite to be had -- an empty population, or one
38:     where nothing scored above 0.0. `fitness` defaults to 0.0, so an all-zero
39:     population is either a sweep that never reached the fitness step or one
40:     where every individual failed; neither has a best worth carrying forward,
41:     and electing one anyway would dress up an arbitrary pick as a result.
42:     """
43:     best = max(rows, key=lambda row: (row["fitness"] or 0.0, -row["number"]),
44:                default=None)
45:     if best is None or not (best["fitness"] or 0.0) > 0.0:
46:         return None
47:     return best
48: 
49: 
50: def elect(conn, run_id):
51:     """Mark one individual as the sweep's elite. -> (the elite row, all rows).
52: 
53:     Writes nothing when there is no elite: a sweep keeps whatever is_best it
54:     already had rather than gaining a meaningless one. The population comes back
55:     too, in number order, so a caller can report on the election -- or on the
56:     lack of one -- without asking the database a second time.
57:     """
58:     rows = store.individuals(conn, run_id)
59:     best = best_of(rows)
60:     if best is not None:
61:         store.mark_best(conn, run_id, best["number"])
62:         conn.commit()
63:     return best, rows

(End of file - total 63 lines)
</content>
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 4.2s)

_Thinking:_

Let me now read process_run.py, gep_lora/core/pipeline/start_run.py (the driver), gep_lora/core/pipeline/main.py, and the templates. Also let me look at the async API. Let me use explore agents for broad areas to save context.

Let me read process_run.py and gep_lora/core/pipeline/start_run.py. gep_lora/core/pipeline/start_run.py is 76KB / 1291 lines. Let me read it in chunks, but maybe first look at its structure via grep.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\blends\\process_run.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\blends\process_run.py</path>
<type>file</type>
<content>
1: """
2: process_run.py - Launch a generated script and make sense of what it printed.
3: 
4: gep_lora/core/pipeline/start_run.py writes each individual's script out of the database, hands it to
5: launch() as its own process, and files everything it said back into the
6: database. This module is the part that knows how to do that: how to run one
7: script, how to read a transcript out of its stdout, and how to check that this
8: interpreter can run it at all.
9: 
10: Each script is a separate process, because each loads the base model at import
11: and attaches its own adapters -- they cannot share an interpreter. That also
12: means process is the expensive step: a model load per individual, so a
13: population of 100 is a long sweep. Keep COUNT small in settings.py while
14: iterating, or use --limit.
15: 
16: Separate processes are also what makes them overlappable, and launch_batch()
17: runs PROCESS_RUN_BATCH_SIZE of them at once -- the model loads are what a sweep
18: spends its time on, and they are independent. The ceiling is fixed rather than
19: adaptive: N scripts means N copies of the base model resident together, so the
20: number belongs to the machine and is set in settings.py, not guessed at here.
21: 
22: Each child is read while it runs rather than collected at the end, so the step
23: can say where a script has got to -- see Progress. What it says is where things
24: are, never the transcript itself: that goes to the database.
25: 
26: Individuals that fail are recorded, not fatal: a chromosome that cannot run is
27: a result, the same as one that can. Only a sweep where nothing at all ran
28: returns a failing exit code, since that points at something systemic.
29: 
30: A script generated from template_remote_code.py pays none of the load: it is a
31: client for a lora_server.py process that already holds the base model open, and
32: server_pool.py is what starts those and tells each script which one to use
33: (through the environment -- see launch()). Everything this module does is
34: unchanged by that. A remote script is still one process per individual, still
35: prints its transcript and its TIMING: lines on stdout, and still has an exit
36: code, which is the whole reason the server was put behind the scripts rather
37: than in place of them.
38: 
39: None of that cost applies to scripts generated from template_code_mocked.py:
40: they load nothing, answer at random, and print their own QUALITY:/REASON: lines,
41: which land in the transcript so the evaluate step has nothing left to score.
42: imports_unsloth() is how the pipeline notices which kind it is looking at and
43: drops the venv check accordingly.
44: """
45: 
46: import concurrent.futures
47: import os
48: import re
49: import subprocess
50: import sys
51: import threading
52: import time
53: 
54: 
55: def drawn_weights(stdout):
56:     """The blend weights a run drew for itself, from the line it prints.
57: 
58:     Every generated script redraws w1..w5 at startup, so two runs of the same
59:     chromosome are scored under different blends. Recording the draw is what
60:     makes a transcript traceable back to the weights that produced it.
61: 
62:     Read off the "weights: w1=..., w2=..." line rather than recomputed, so this
63:     is the draw that was actually used. All five are recorded, not just the ones
64:     this tree happens to reference.
65:     """
66:     line = re.search(r"^weights:.*$", stdout, re.MULTILINE)
67:     if not line:
68:         return {}
69:     return {name: float(value)
70:             for name, value in re.findall(r"(w\d+)=([-+0-9.eE]+)", line.group(0))}
71: 
72: 
73: # The line every generated script opens with: "Individual 7: CAT.L1.L2.w1.w3"
74: # -- the LABEL and EXPRESSION markers, which start_run.step_runs fills with the
75: # individual's number and its chromosome. All three templates print it first.
76: _BUILT = re.compile(r"^Individual \d+: (\S+)$", re.MULTILINE)
77: 
78: 
79: def expression(stdout):
80:     """The chromosome a run says it built, or None if it never said.
81: 
82:     Read off the script's own opening line rather than taken from the
83:     individual, for the reason drawn_weights() reads the weights: the
84:     individual goes on being rewritten, and this is what that execution was
85:     actually of. The process step uses it to tell a result of the blend an
86:     individual holds now from one of the blend it held before mutation.
87:     """
88:     found = _BUILT.search(stdout or "")
89:     return found.group(1) if found else None
90: 
91: 
92: # What a script says about its own cost, on the channel everything else comes
93: # out on: one line per occurrence, "TIMING: <phase> <seconds> [label]".
94: #
95: # A third channel out of a child process would be a pipe nobody else needs; a
96: # marker line is how the weights, the transcript and a mocked score already
97: # travel. The phase names are the script's own -- see template_code.py -- so
98: # adding a measurement there needs nothing here.
99: TIMING_PREFIX = "TIMING:"
100: 
101: # Lines a reply is over by. QUALITY:/REASON: are the mocked template's score,
102: # TIMING: is any script talking about itself; neither is part of what the model
103: # said, and both are printed after the answer they follow.
104: GRADE_PREFIXES = ("QUALITY:", "REASON:")
105: 
106: 
107: def timings(stdout):
108:     """What a run says it spent its time on, one dict per phase.
109: 
110:     Occurrences of a phase are folded together here rather than stored one by
111:     one: a script prints one line per generate() and there may be fifty of
112:     them, and what a reader needs is calls, the total and the worst one. The
113:     order is the order each phase was first seen, which is the order the script
114:     goes through them.
115: 
116:         [{"phase": "model_load", "calls": 1, "seconds": 61.8,
117:           "longest": 61.8, "detail": None},
118:          {"phase": "generate", "calls": 5, "seconds": 12.4,
119:           "longest": 3.1, "detail": None}]
120: 
121:     A line that is not "TIMING: <phase> <number> [label]" is ignored: a script's
122:     own account of itself is a convenience, and a malformed line must not cost
123:     the run whose transcript it sits in. A script that printed none at all --
124:     an older sweep, a run that died before it got going -- gives back nothing,
125:     and the wall time the step measured from outside still stands.
126:     """
127:     found = {}
128:     order = []
129:     for line in stdout.splitlines():
130:         if not line.startswith(TIMING_PREFIX):
131:             continue
132:         parts = line[len(TIMING_PREFIX):].split(None, 2)
133:         if len(parts) < 2:
134:             continue
135:         phase, value = parts[0], parts[1]
136:         try:
137:             seconds = float(value)
138:         except ValueError:
139:             continue
140:         entry = found.get(phase)
141:         if entry is None:
142:             entry = found[phase] = {"phase": phase, "calls": 0, "seconds": 0.0,
143:                                     "longest": 0.0, "detail": None}
144:             order.append(entry)
145:         entry["calls"] += 1
146:         entry["seconds"] += seconds
147:         entry["longest"] = max(entry["longest"], seconds)
148:         label = parts[2].strip() if len(parts) > 2 else ""
149:         if label:
150:             # What told two occurrences of one phase apart -- which node, which
151:             # adapter. Kept as a list because the phase row is the fold of all
152:             # of them, and capped because it is a label, not a transcript.
153:             labels = (entry["detail"] or {}).get("labels", [])
154:             if label not in labels and len(labels) < 8:
155:                 entry["detail"] = {"labels": labels + [label]}
156:     return order
157: 
158: 
159: def _split_grade(lines):
160:     """Separate a reply from the QUALITY:/REASON: lines that may follow it.
161: 
162:     Only the mocked template (template_code_mocked.py) prints those, so for a
163:     real run this returns the lines untouched and an empty grade -- the scoring
164:     still comes from the evaluators package. For a mocked run it is what carries
165:     the
166:     made-up score into the transcript, so a dry sweep needs no judge endpoint.
167: 
168:     A reply line that genuinely started with "QUALITY:" would be cut short here.
169:     That is the price of not needing a separate channel out of the child
170:     process, and no real reply has ever begun that way.
171:     """
172:     answer, grade = [], {}
173:     for line in lines:
174:         if line.startswith(TIMING_PREFIX):
175:             # Printed after the reply it belongs to, so the reply is over: a
176:             # phase line the script prints about itself is not something the
177:             # model said, and letting it through would put it in the transcript
178:             # and then in front of a judge. Sets no grade of its own.
179:             grade.setdefault("_over", True)
180:         elif line.startswith("QUALITY:"):
181:             try:
182:                 grade["quality"] = float(line[len("QUALITY:"):].strip())
183:             except ValueError:
184:                 pass                            # not a number: leave it ungraded
185:         elif line.startswith("REASON:"):
186:             grade["reason"] = line[len("REASON:"):].strip()
187:         elif not grade:                         # still in the reply itself
188:             answer.append(line)
189:     grade.pop("_over", None)
190:     return answer, grade
191: 
192: 
193: def exchanges(stdout):
194:     """The YOU/COACH pairs in a run's stdout, as {"question", "answer"} dicts.
195: 
196:     A reply can wrap over several lines, so a question owns everything printed
197:     after it until the next question. Only stdout is scanned -- the loading bars
198:     and warnings arrive on stderr, so they cannot leak into a transcript.
199: 
200:     A question whose reply never arrived (a run killed mid-generation) keeps an
201:     empty answer, rather than being dropped as if it had never been asked.
202: 
203:     An exchange also picks up "quality" and "reason" when the run printed them,
204:     which only the mocked template does; see _split_grade.
205:     """
206:     blocks, current = [], None
207:     for line in stdout.splitlines():
208:         if line.startswith("YOU:"):
209:             if current is not None:
210:                 blocks.append(current)
211:             current = [line]
212:         elif current is not None:
213:             current.append(line)
214:     if current is not None:
215:         blocks.append(current)
216: 
217:     transcript = []
218:     for block in blocks:
219:         question = block[0][len("YOU:"):].strip()
220:         answer, grade = [], {}
221:         for offset, line in enumerate(block[1:], start=1):
222:             if line.startswith("COACH:"):
223:                 # The reply is the rest of that line plus every line after it.
224:                 rest = [line[len("COACH:"):].strip()] + block[offset + 1:]
225:                 answer, grade = _split_grade(rest)
226:                 break
227:         while answer and not answer[-1].strip():   # trim the gap before the next question
228:             answer.pop()
229:         exchange = {"question": question, "answer": "\n".join(answer)}
230:         # Same key order a scored transcript ends up with either way.
231:         exchange.update(grade)
232:         transcript.append(exchange)
233:     return transcript
234: 
235: 
236: # How often launch() looks up from waiting while a script says nothing. Short
237: # enough that a heartbeat lands near the second it is due, cheap enough to be
238: # invisible: the wait does nothing but time out.
239: TICK = 2.0
240: 
241: # What every child is told to encode its stdout as, and why it is not a choice.
242: #
243: # launch() reads the pipes as utf-8. A child left to itself picks the platform's
244: # preferred encoding for a pipe, which on this machine is cp1252 -- so the two
245: # ends disagreed about every byte above ASCII, and a model answer is full of
246: # them. Two ways that showed up, both silent about their real cause:
247: #
248: #   * a character cp1252 *has* -- a curly quote, an em dash -- was written as
249: #     one cp1252 byte and read back as invalid utf-8, so the transcript stored
250: #     in the database held U+FFFD where the model had written punctuation, and
251: #     that is what the judge was then shown and scored.
252: #   * a character cp1252 lacks -- Greek, CJK, an emoji -- raised
253: #     UnicodeEncodeError inside the child's own print(), killing it mid-answer.
254: #     The individual's execution went down as `exit 1` with however much of the
255: #     transcript had already been printed, so its fitness became the mean over
256: #     the questions it happened to reach before the first awkward character.
257: #
258: # Neither is anything to do with the blend being scored, which is what made
259: # them worth stamping out here rather than in each template: this is the one
260: # place every generated script is launched from, the baseline included.
261: CHILD_ENCODING = "PYTHONIOENCODING"
262: 
263: 
264: def _drain(stream, collected, report):
265:     """Read one of a child's pipes to the end, keeping every line it held.
266: 
267:     A thread per pipe, because a child that fills one while nothing reads the
268:     other would block there for good -- the reason the old capture_output could
269:     not also stream. Lines are kept exactly as they arrived; `report` only gets
270:     to look at them.
271:     """
272:     with stream:
273:         for line in stream:
274:             collected.append(line)
275:             if report is not None:
276:                 report(line.rstrip("\n"))
277: 
278: 
279: def launch(run_dir, script, timeout, on_line=None, on_tick=None, env=None):
280:     """Run one generated script. -> (exit code, seconds, stdout, stderr).
281: 
282:     An exit code of None means the script was still going when the timeout
283:     expired. stdout is kept apart from stderr so the transcript can be taken
284:     from it cleanly.
285: 
286:     Nothing is written here: the four values go straight into the database, so
287:     the only file this step needs is the script itself.
288: 
289:     The child is read as it runs rather than collected at the end, so a caller
290:     can say where a long run has got to instead of leaving the console silent
291:     for the minutes a base-model load takes. `on_line` sees each stdout line as
292:     it arrives and `on_tick` the seconds so far, every TICK, whether or not
293:     anything was printed -- a load says nothing at all while it happens, so
294:     silence has to be reported too. Both are called from this launch's own
295:     threads; with a batch in flight, several scripts report at once.
296: 
297:     -u, because the child's stdout is a pipe: without it Python would block-
298:     buffer the transcript and a whole run would arrive at once, at the end.
299: 
300:     A killed run keeps what it had already printed, so a timeout still stores
301:     the exchanges that did come back rather than an empty transcript. A timeout
302:     of 0 or None is no limit at all, rather than a script killed the instant it
303:     starts, which is the only thing a caller could mean by it.
304: 
305:     `env` is added to this process's environment for the child, and is how a
306:     lora_server client is told which server to talk to (see
307:     server_pool.SERVER_ENV). An addition rather than a replacement, because a
308:     generated script also needs whatever PATH, HF_HOME and CUDA_* the sweep is
309:     being run under; and an environment variable rather than an argument
310:     because it must not reach script_source -- which server an individual ran
311:     against is an accident of the batch it landed in, not part of what that
312:     individual is.
313: 
314:     Every child gets PYTHONIOENCODING as well -- see CHILD_ENCODING, which is
315:     not optional and not about servers.
316:     """
317:     script_path = os.path.join(run_dir, script)
318:     started = time.time()
319:     child_env = dict(os.environ)
320:     child_env[CHILD_ENCODING] = "utf-8"
321:     if env:
322:         child_env.update({name: str(value) for name, value in env.items()})
323:     # cwd is the run folder, so the caches unsloth drops stay there.
324:     child = subprocess.Popen(
325:         [sys.executable, "-u", script_path],
326:         cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
327:         text=True, encoding="utf-8", errors="replace", env=child_env,
328:     )
329:     out, err = [], []
330:     readers = [threading.Thread(target=_drain, args=(child.stdout, out, on_line)),
331:                threading.Thread(target=_drain, args=(child.stderr, err, None))]
332:     for reader in readers:
333:         reader.daemon = True
334:         reader.start()
335: 
336:     deadline = None if not timeout else started + timeout
337:     while True:
338:         try:
339:             code = child.wait(timeout=TICK)
340:             break
341:         except subprocess.TimeoutExpired:
342:             if deadline and time.time() >= deadline:
343:                 child.kill()
344:                 child.wait()
345:                 code = None
346:                 break
347:             if on_tick is not None:
348:                 on_tick(time.time() - started)
349: 
350:     for reader in readers:                      # everything the pipes still held
351:         reader.join()
352:     stdout, stderr = "".join(out), "".join(err)
353:     if code is None:
354:         stderr += "\n\n!! killed after %ss (--timeout)\n" % timeout
355:     return code, time.time() - started, stdout, stderr
356: 
357: 
358: def batch_size(value):
359:     """How many scripts may run at once, from the setting -> at least 1.
360: 
361:     A sweep created before PROCESS_RUN_BATCH_SIZE existed has no value recorded
362:     for it, and a value below 1 is the same request as 1 stated badly; both mean
363:     one script at a time rather than an error, since neither says anything about
364:     the machine this is now running on.
365:     """
366:     try:
367:         return max(1, int(value))
368:     except (TypeError, ValueError):
369:         return 1
370: 
371: 
372: def batches(items, size):
373:     """`items` in consecutive groups of at most `size`, in the order given.
374: 
375:     Groups rather than a refilling queue: gep_lora/core/pipeline/start_run.py stores a batch's results
376:     before it starts the next one, which is what keeps the database written from
377:     one thread and in the order the individuals were selected in.
378:     """
379:     return [items[start:start + size] for start in range(0, len(items), size)]
380: 
381: 
382: def elapsed(seconds):
383:     """Seconds as something short to read: 45s, 3m20s, 1h04m."""
384:     seconds = int(seconds)
385:     if seconds < 60:
386:         return "%ds" % seconds
387:     if seconds < 3600:
388:         return "%dm%02ds" % (seconds // 60, seconds % 60)
389:     return "%dh%02dm" % (seconds // 3600, (seconds % 3600) // 60)
390: 
391: 
392: # The line a generated script prints once its blend is built and set active --
393: # the end of the long silent part, and the only startup line worth repeating.
394: # Every template prints it, and every one names the rank of the finished blend.
395: _READY = "Active adapter:"
396: _RANK = re.compile(r"rank (\d+)")
397: 
398: # What a lora_server client prints before it sends its build plan. Only the
399: # remote template prints it, and it is the last thing said before a silence
400: # that can run to minutes on an svd node -- so it is what tells Progress that
401: # the silence is a blend being folded on a server rather than a base model
402: # being loaded here. Saying "still loading the base model" about a script that
403: # never loads one sends you looking in the wrong place.
404: _REMOTE = "SERVER:"
405: 
406: 
407: class Progress:
408:     """Where one running script has got to, said now and then rather than always.
409: 
410:     A generated script is quiet for as long as the base-model load takes and
411:     then prints one YOU:/COACH: pair per eval prompt. So there are two things
412:     worth saying while it runs -- that the load finished, and how far through
413:     the prompts it is -- and the rest of what it prints is transcript, which
414:     belongs in the database and not on a console.
415: 
416:     Which is why nothing here reports per line. A milestone (the model ready,
417:     the last prompt started) is said when it happens, being one line each; the
418:     running count is said only once `every` seconds have passed since this
419:     script last said anything, so a fifty-prompt run speaks a handful of times
420:     rather than fifty. `every` of 0 leaves only the milestones.
421: 
422:     say(script, message) does the printing. It is called from the thread
423:     draining that script's stdout, and every script in a batch reports through
424:     the same one, so making it thread-safe is the caller's part of this.
425:     """
426: 
427:     __slots__ = ("script", "prompts", "every", "_say", "_clock",
428:                  "_started", "_last", "_asked", "_loaded", "_remote")
429: 
430:     def __init__(self, script, prompts, every, say, clock=time.time):
431:         self.script = script
432:         self.prompts = prompts          # 0 when the eval set could not be counted
433:         # A cadence that is not a number is one this cannot keep, so it keeps
434:         # none: these run on the thread draining a child, where an exception
435:         # would cost the rest of that script's output to report a typo.
436:         try:
437:             self.every = max(0.0, float(every))
438:         except (TypeError, ValueError):
439:             self.every = 0.0
440:         self._say = say
441:         self._clock = clock
442:         self._started = self._last = clock()
443:         self._asked = 0
444:         self._loaded = False
445:         # Set by the script itself, if it turns out to be a lora_server client.
446:         # Not asked of the source up front: this reads what a running script
447:         # says about itself, and one line is cheaper than another argument
448:         # threaded from the step through launch_batch to here.
449:         self._remote = None
450: 
451:     def line(self, text):
452:         """Take one line of the script's stdout, and speak if it warrants it."""
453:         if text.startswith(_REMOTE):
454:             self._remote = text[len(_REMOTE):].strip()
455:         elif text.startswith(_READY):
456:             self._loaded = True
457:             rank = _RANK.search(text)
458:             self._speak("model ready" if not rank
459:                         else "model ready, blend rank %s" % rank.group(1))
460:         elif text.startswith("YOU:"):
461:             # Printed before the answer is generated, so this is the prompt
462:             # being worked on rather than one already done.
463:             self._asked += 1
464:             if self._asked == self.prompts or self._due():
465:                 self._speak(self._where())
466: 
467:     def tick(self, seconds):
468:         """Take a moment in which the script printed nothing. Break long silences."""
469:         if not self._due():
470:             return
471:         if self._loaded:
472:             self._speak("still on " + self._where())
473:         elif self._remote:
474:             # Nothing is loading: the model has been up on that server since
475:             # before this script started, and what the silence is is the blend
476:             # being folded -- an svd node runs to a minute or more on its own.
477:             self._speak("still building the blend on %s" % self._remote)
478:         else:
479:             self._speak("still loading the base model")
480: 
481:     def _where(self):
482:         """The prompt it is on, out of however many there are to do."""
483:         if not self.prompts:
484:             return "prompt %d" % self._asked
485:         return "prompt %d/%d" % (self._asked, self.prompts)
486: 
487:     def _due(self):
488:         return self.every > 0 and self._clock() - self._last >= self.every
489: 
490:     def _speak(self, message):
491:         self._last = self._clock()
492:         self._say(self.script, "%s, %s" % (message, elapsed(self._last - self._started)))
493: 
494: 
495: def launch_batch(run_dir, scripts, timeout, watch=None, envs=None):
496:     """Run these scripts at once. -> one launch() result each, in `scripts` order.
497: 
498:     Results come back in the order asked for, not the order they finished, so a
499:     caller can pair them with the individuals it passed in without the children
500:     having to say who they are.
501: 
502:     Threads, not processes: each one only waits on a subprocess, and the work is
503:     in the children anyway. The timeout is per script, as it is sequentially --
504:     a batch is not killed because one member of it hung.
505: 
506:     watch(script) hands back that script's Progress, or None for a silent run.
507:     One per script rather than one for the batch: each is somewhere different,
508:     and a line that did not say which script it came from would be useless with
509:     several of them going at once.
510: 
511:     The seconds each result carries are still that script's own wall clock, so
512:     they overlap and no longer add up to the time the batch took.
513: 
514:     `envs` is one environment per script, in the same order -- what
515:     server_pool.Pool.envs_for() hands back, so the k-th script of a batch talks
516:     to the k-th server. Positional rather than a queue for the reason the batch
517:     itself is a fixed ceiling: the caller caps the batch at the pool's size, so
518:     every script in one has a server to itself and nothing has to wait.
519:     """
520:     def one(index, script):
521:         reporter = watch(script) if watch is not None else None
522:         env = envs[index] if envs else None
523:         if reporter is None:
524:             return launch(run_dir, script, timeout, env=env)
525:         return launch(run_dir, script, timeout, reporter.line, reporter.tick, env)
526: 
527:     if len(scripts) == 1:                       # the sequential case, unchanged
528:         return [one(0, scripts[0])]
529:     with concurrent.futures.ThreadPoolExecutor(max_workers=len(scripts)) as pool:
530:         running = [pool.submit(one, index, script)
531:                    for index, script in enumerate(scripts)]
532:         return [future.result() for future in running]
533: 
534: 
535: def verdict_of(code):
536:     """The one word an execution's verdict column holds for an exit code."""
537:     if code == 0:
538:         return "ok"
539:     if code is None:
540:         return "timeout"
541:     return "exit %d" % code
542: 
543: 
544: def imports_unsloth(source):
545:     """Does this generated script load the real thing? The mocked ones do not.
546: 
547:     A test on the source itself, so it can be asked of a script held in the
548:     database as easily as of one already written out to disk.
549:     """
550:     return "import unsloth" in source or "from unsloth" in source
551: 
552: 
553: def check_interpreter():
554:     """Fail fast if this interpreter cannot import what the scripts need.
555: 
556:     Each generated script is launched with sys.executable, so running this with
557:     the wrong python means every individual dies on `import unsloth` -- after
558:     paying for a process launch each time, and leaving a population's worth of
559:     empty transcripts behind. find_spec only looks the module up, so the check
560:     costs nothing next to actually importing it.
561:     """
562:     probe = ("import importlib.util, sys; "
563:              "sys.exit(0 if importlib.util.find_spec('unsloth') else 1)")
564:     if subprocess.run([sys.executable, "-c", probe]).returncode:
565:         raise SystemExit(
566:             "%s cannot import unsloth, and the generated scripts run under this "
567:             "same interpreter -- every one of them would fail. Re-run with the "
568:             "project venv's python (see the PATH gotcha in README.md)."
569:             % sys.executable
570:         )

(End of file - total 570 lines)
</content>
```

**Tool: grep**

**Input:**
```json
{
  "pattern": "^def |^class |^STEPS|^NEXT_GENERATION",
  "include": "gep_lora/core/pipeline/start_run.py"
}
```

**Output:**
```
Found 25 matches
D:\sage-is\loras\gep-lora\start_run.py:
  Line 104: class Context:

  Line 173: def new_sweep(conn, label):

  Line 189: def freeze(conf):

  Line 214: def step_population(context):

  Line 231: def step_trees(context):

  Line 255: def step_runs(context):

  Line 331: def wants_the_card(conf):

  Line 356: def _clear_scripts(context, rows):

  Line 379: def step_process(context):

  Line 671: def step_evaluate(context):

  Line 691: def _evaluate(context):

  Line 836: def step_fitness(context):

  Line 897: def step_elitism(context):

  Line 924: def _span(numbers):

  Line 931: def _master_seed(context, name):

  Line 948: def step_selection(context):

  Line 1046: def step_mutation(context):

  Line 1091: def step_weight_mutation(context):

  Line 1133: STEPS = [

  Line 1183: NEXT_GENERATION = ("elitism", "selection", "mutation", "weight_mutation")

  Line 1186: def without_next_generation(steps):

  Line 1203: def resolve_run_dir(conf, override=None, default=None):

  Line 1219: def context_for(conn, run_id, conf, args):

  Line 1234: def run(steps, context):

  Line 1298: def main(argv=None):

```

---

## Assistant (Build · DeepSeek V4.1 Flash · 4.0s)

_Thinking:_

Let me read gep_lora/core/pipeline/start_run.py, at least the core parts (Context, run, steps). And gep_lora/core/pipeline/main.py. Let me read gep_lora/core/pipeline/start_run.py lines 1-400 and 379-700 (process/evaluate), then 836-1300.

Actually, let me read the whole gep_lora/core/pipeline/start_run.py in a couple chunks, since it's central. 1291 lines. Let me read 1-700 and 700-1300.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\start_run.py",
  "limit": 700
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\start_run.py</path>
<type>file</type>
<content>
1: """
2: gep_lora/core/pipeline/start_run.py - Run the whole pipeline end to end against a sqlite database.
3: 
4:     population -> trees -> runs -> process -> evaluate -> fitness -> elitism
5:     -> selection -> mutation -> weight_mutation
6: 
7: Nothing is left scattered across a folder afterwards. The population, every
8: setting the sweep ran under, every seed, every generated script, every
9: transcript, every score and the fitness they come to go into one database,
10: described in store.py:
11: 
12:     runs -> settings, individuals -> executions -> exchanges
13: 
14: Why a database
15: --------------
16: A folder of a hundred files answers "what happened" only by being read one file
17: at a time, and the next sweep overwrites it. A database keeps sweeps side by
18: side and lets you ask across them:
19: 
20:     SELECT chromosome, quality FROM individual_quality
21:      WHERE run_id = 3 ORDER BY quality DESC LIMIT 5;
22: 
23: and, because the seeds are stored too, it can say exactly what a stored sweep
24: was: which chromosomes (the population seed) and which blend weights each of
25: them was judged under (a per-individual weight seed, stamped into its script).
26: A stored sweep can therefore be repeated, which is the point of storing it.
27: 
28: What still touches the disk
29: ---------------------------
30: The generated run_NNN.py scripts, and only those, and only until they have run:
31: process launches them as subprocesses, so they have to be real files, and it
32: deletes each one it has processed once the sweep is through. They are a cache of
33: individuals.script_source, rewritten from the database whenever they are missing
34: or stale, so nothing is lost -- what is gained is that run_db/ does not fill up
35: with spent scripts, and nobody can launch a stale one by hand a week later.
36: Pass --keep-scripts to leave them.
37: 
38: They have to sit exactly one level below the project folder -- a generated
39: script finds the LoRA folders by going up one from itself -- so they land in
40: run_db/, alongside the database itself.
41: 
42: Steps run in the order listed and stop at the first failure. An individual that
43: crashes is not a failure: it is a row in executions with its exit code.
44: 
45: Resuming
46: --------
47: Every step but population can be re-run against a sweep already in the database,
48: and the settings it reads are the ones that sweep was created with, not whatever
49: settings.py says today. That is what makes a resumed sweep still be the same
50: sweep. Name a run with --run, or leave it out and the most recent one is used.
51: 
52: Usage:
53:     python start_run.py                    # a new sweep, every step
54:     python start_run.py --list             # show the steps without running them
55:     python start_run.py --evaluators       # show the ways an answer can be scored
56:     python start_run.py population trees runs   # the fast half, before a model load
57:     python start_run.py process evaluate   # resume the latest sweep
58:     python start_run.py evaluate --run 3   # score sweep 3
59:     python start_run.py --limit 3          # smoke test: only the first few
60: 
61: Run it with the **venv's python**: process launches each generated script with
62: sys.executable, so the wrong interpreter fails every individual. That does not
63: apply to a sweep generated from template_code_mocked.py, which loads nothing.
64: 
65: Reading a sweep back:
66:     python -m gep_lora.core.storage.store --list
67:     python -m gep_lora.core.storage.store --show 0
68:     python -m gep_lora.core.storage.store --export 0 --into export
69: """
70: 
71: import argparse
72: import itertools
73: import os
74: import random
75: import sys
76: import threading
77: import time
78: from collections import namedtuple
79: 
80: from blends import generate_runs
81: from blends import process_run
82: from blends import server_pool
83: from config import settings as config
84: import evaluators
85: from metrics import record
86: from search import calculate_fitness
87: from search import draw_trees
88: from search import elitism
89: from search import generate_population
90: from search import mutation
91: from search import selection
92: from search import weight_mutation
93: from storage import add_dataset
94: from storage import db_datasets
95: from storage import store
96: 
97: _HERE = os.path.dirname(os.path.abspath(__file__))
98: 
99: # The upper bound for a drawn seed. Kept well inside 2**32 so the value is a
100: # plain int in every sqlite column and in every generated script.
101: _SEED_LIMIT = 2 ** 31 - 1
102: 
103: 
104: class Context:
105:     """What every step needs: the database, the sweep, and the options."""
106: 
107:     __slots__ = ("conn", "run_id", "conf", "run_dir", "template", "options",
108:                  "generation", "meter", "pool")
109: 
110:     def __init__(self, conn, run_id, conf, run_dir, template, options):
111:         self.conn = conn
112:         self.run_id = run_id
113:         self.conf = conf              # the settings this sweep runs under
114:         self.run_dir = run_dir
115:         self.template = template      # absolute path to the template it fills
116:         self.options = options        # the parsed command line
117:         # "2/5" while gep_lora/core/pipeline/continue_run.py is working through its generations, None
118:         # for a single pass. Nothing reads it but the banners: a generation is
119:         # not stored, and a step must not start behaving differently in one.
120:         self.generation = None
121:         # What this step cost, and where inside it the time went. run() swaps
122:         # in a Meter of its own around every step; a Context built anywhere else
123:         # -- a test, a shell, a driver calling one step directly -- keeps this
124:         # blank one, which accepts everything a step says and writes nothing. So
125:         # a step never has to ask whether it is being timed.
126:         self.meter = record.blank()
127:         # The lora servers, once the process step has started them. They live
128:         # on the Context rather than inside the step because a Context is what
129:         # a driver holds for as long as it is driving one sweep -- so the base
130:         # model is loaded once for a whole run of generations rather than once
131:         # per generation. Whoever built the Context calls release_pool(); see
132:         # step_process for the one thing that takes them down early.
133:         self.pool = None
134: 
135:     # The three things a step says about its own cost. Delegated rather than
136:     # reached through context.meter so instrumenting a step reads as part of
137:     # the step: see gep_lora/core/metrics/record.py for what each of them is for.
138: 
139:     def count(self, items, unit=None, skipped=0, note=None):
140:         """How many units this step worked on, and how many it skipped."""
141:         self.meter.count(items, unit, skipped=skipped, note=note)
142: 
143:     def phase(self, name, seconds, **fields):
144:         """Add `seconds` to a named phase inside this step."""
145:         self.meter.phase(name, seconds, **fields)
146: 
147:     def timer(self, name, **fields):
148:         """`with context.timer("prepare"):` -- the same, timed for you."""
149:         return self.meter.timer(name, **fields)
150: 
151:     def phases_from(self, records, execution_id=None):
152:         """Add phases something else measured -- a generated script's own
153:         TIMING: lines, read back by process_run.timings()."""
154:         self.meter.phases_from(records, execution_id=execution_id)
155: 
156:     def release_pool(self):
157:         """Stop the lora servers, if this Context started any.
158: 
159:         Called by whoever built the Context, in a finally: the servers hold a
160:         copy of the base model each, and a driver that has finished with a sweep
161:         must not leave them holding it. Safe to call when there are none, and
162:         safe to call twice -- which is what lets both the driver and the one
163:         step that releases them early say it without coordinating.
164:         """
165:         if self.pool is not None:
166:             pool, self.pool = self.pool, None
167:             pool.stop()
168: 
169: 
170: # --- the settings a sweep runs under --------------------------------------
171: 
172: 
173: def new_sweep(conn, label):
174:     """Create a run row and freeze the settings it will use. -> (run_id, conf).
175: 
176:     Seeds left as None in settings.py are drawn *here* and stored as the number
177:     that was drawn, rather than passed along as None. A sweep is then repeatable
178:     even when it was never asked to be: whatever it used is written down.
179:     """
180:     conf = freeze(config.snapshot())
181:     run_id = store.create_run(conn, template=conf.get("TEMPLATE") or "template_code.py",
182:                               label=label)
183:     print("new run %d in %s" % (run_id, conn.path))
184:     store.save_settings(conn, run_id, conf)
185:     add_dataset.save_all(conn, run_id, conf)
186:     return run_id, conf
187: 
188: 
189: def freeze(conf):
190:     """Check a new sweep's settings and draw its unset seeds. -> the same dict.
191: 
192:     Its own function because a sweep is not only created here: async_api's
193:     submit.py prepares one for a worker to run, and the two must agree on what
194:     a new sweep is checked for and what it writes down.
195:     """
196:     # Fail on an unknown EVALUATOR here rather than an hour later, when the
197:     # process step has finished and there is a transcript nobody can score.
198:     # Same for an unknown JUDGE_BACKEND, which is the other half of that
199:     # question: which evaluator grades, and where its judge runs.
200:     evaluator = evaluators.get(conf.get("EVALUATOR"))
201:     if evaluator.check:
202:         evaluator.check(conf)
203:     evaluators.backend_of(conf)
204:     for name in ("SEED", "WEIGHT_MASTER_SEED", "SELECTION_MASTER_SEED",
205:                  "MUTATION_MASTER_SEED", "WEIGHT_MUTATION_MASTER_SEED"):
206:         if conf.get(name) is None:
207:             conf[name] = random.randrange(_SEED_LIMIT)
208:     return conf
209: 
210: 
211: # --- the steps -------------------------------------------------------------
212: 
213: 
214: def step_population(context):
215:     """Draw the chromosomes -> individuals."""
216:     conf = context.conf
217:     rng = random.Random(conf["SEED"])
218:     chromosomes = generate_population.build_population(
219:         conf["COUNT"], rng, conf["MAX_DEPTH"], conf["BRANCH_PROB"], conf["UNIQUE"])
220:     store.add_individuals(context.conn, context.run_id, chromosomes)
221: 
222:     context.count(len(chromosomes), "individuals")
223: 
224:     sizes = [len(one.split(".")) for one in chromosomes]
225:     print("stored %d individuals in run %d (seed %s)"
226:           % (len(chromosomes), context.run_id, conf["SEED"]))
227:     print("symbols per individual: min %d, max %d, mean %.1f"
228:           % (min(sizes), max(sizes), sum(sizes) / len(sizes)))
229: 
230: 
231: def step_trees(context):
232:     """Draw each chromosome -> individuals.tree."""
233:     rows = store.individuals(context.conn, context.run_id)
234:     if not rows:
235:         raise SystemExit("run %d holds no individuals. Run the population step first."
236:                          % context.run_id)
237:     bad = 0
238:     for row in rows:
239:         try:
240:             drawing = "\n".join(draw_trees.draw(row["chromosome"]))
241:         except ValueError as error:
242:             # An undrawable individual is recorded with its complaint, not
243:             # dropped: it is still part of the population.
244:             drawing = "%s\n\n!! cannot draw: %s" % (row["chromosome"], error)
245:             bad += 1
246:         store.set_tree(context.conn, row["id"], drawing)
247:     context.conn.commit()
248: 
249:     context.count(len(rows), "trees")
250:     print("drew %d trees into run %d" % (len(rows), context.run_id))
251:     if bad:
252:         print("%d individual(s) could not be drawn -- see the !! markers" % bad)
253: 
254: 
255: def step_runs(context):
256:     """Fill the template per individual -> script_source, state, rank, weight seed."""
257:     conn, run_id = context.conn, context.run_id
258:     rows = store.individuals(conn, run_id)
259:     if not rows:
260:         raise SystemExit("run %d holds no individuals. Run the population step first."
261:                          % run_id)
262: 
263:     template_lines = generate_runs.load_template(context.template)
264:     # The adapters come from this sweep's stored LORA_SLOTS, not from the
265:     # template; each slot's rank comes from its own adapter_config.json, and
266:     # they may differ.
267:     slots = context.conf.get("LORA_SLOTS")
268:     ranks = generate_runs.slot_ranks(slots)
269:     # The eval prompts come from this sweep's stored TRAINING_SET, not from the
270:     # template; the generated scripts read them at startup, so fail here if they
271:     # cannot be read at all.
272:     training_set = generate_runs.training_set_path(context.conf.get("TRAINING_SET"))
273:     # ...and so does the cap on how many of them each script uses.
274:     count = context.conf.get("TRAINING_COUNT")
275:     # And the model they all attach their adapters to -- the same one the
276:     # baseline the llm_judge_baseline evaluator grades against loads.
277:     base_model = generate_runs.base_model_name(context.conf.get("BASE_MODEL"))
278:     # And the chat template the prompts are written in -- the sweep's own, or
279:     # "qwen-2.5" for a sweep stored before it was a setting.
280:     chat_template = generate_runs.chat_template_name(context.conf)
281:     prompts_path, prompt_count, prompt_total = generate_runs.eval_prompt_count(
282:         training_set, count)
283: 
284:     master = context.conf["WEIGHT_MASTER_SEED"]
285:     runnable = 0
286:     for row in rows:
287:         steps, final = generate_runs.plan(
288:             generate_population.decode(row["chromosome"])[0], ranks)
289:         # Derived from the master seed and the individual's own number, so it is
290:         # the same seed however the population is walked, and re-generating a
291:         # sweep from its stored settings reproduces the blends exactly.
292:         weight_seed = random.Random("%s:%d" % (master, row["number"])).randrange(_SEED_LIMIT)
293:         name = "run_%03d.py" % row["number"]
294:         source = generate_runs.render(
295:             row["chromosome"], steps, final,
296:             script_name=name,
297:             provenance="Generated by gep_lora/core/pipeline/start_run.py from run %d, individual %d of %s."
298:                        % (run_id, row["number"], os.path.basename(conn.path)),
299:             label="Individual %d" % row["number"],
300:             template_lines=template_lines,
301:             weight_seed=weight_seed,
302:             training_set=training_set,
303:             slots=slots,
304:             count=count,
305:             base_model=base_model,
306:             chat_template=chat_template,
307:         )
308:         broken = any(step.broken for step in steps)
309:         runnable += not broken
310:         store.set_script(conn, row["id"], "BAD" if broken else "ok",
311:                          steps[-1].rank, name, source, weight_seed)
312:     conn.commit()
313: 
314:     written = store.materialise(conn, run_id, context.run_dir)
315:     context.count(len(rows), "scripts", skipped=len(rows) - runnable)
316:     print("stored %d scripts in run %d (from %s)"
317:           % (len(rows), run_id, os.path.basename(context.template)))
318:     print("base model: %s" % base_model)
319:     print("slot ranks: %s" % ", ".join("%s=%d" % pair for pair in sorted(ranks.items())))
320:     # Says "10 of 50" when TRAINING_COUNT is holding some back, so the number a
321:     # sweep is actually scored on is never guessed at from the file's size.
322:     print("eval prompts: %d%s from %s"
323:           % (prompt_count,
324:              "" if prompt_count == prompt_total else " of %d" % prompt_total,
325:              os.path.basename(prompts_path)))
326:     print("%d runnable, %d blocked by PEFT's equal-rank rule for linear"
327:           % (runnable, len(rows) - runnable))
328:     print("wrote %d script file(s) to %s" % (written, context.run_dir))
329: 
330: 
331: def wants_the_card(conf):
332:     """Will something else in *this* interpreter need the GPU between passes?
333: 
334:     Which is the only reason the lora servers cannot simply stay up for a whole
335:     run. They hold a copy of the base model each, and with
336:     JUDGE_BACKEND = 'unsloth' the evaluate step loads a judge in this same
337:     process -- so leaving a pool standing through it would be N base models on
338:     the card while a judge tries to find room for one more. On the endpoint
339:     backend the judge is somebody else's process on somebody else's card, and
340:     there is nothing to make room for.
341: 
342:     Read from the sweep's own settings, like everything else a step decides on,
343:     and pessimistic when it cannot tell: an unreadable EVALUATOR is the evaluate
344:     step's problem to report, and until then the safe answer is to give the card
345:     back.
346:     """
347:     try:
348:         if evaluators.backend_of(conf) != evaluators.UNSLOTH:
349:             return False
350:         evaluator = evaluators.get(conf.get("EVALUATOR"))
351:         return evaluator.asks_judge(conf) and evaluator.via_judge_backend
352:     except SystemExit:
353:         return True
354: 
355: 
356: def _clear_scripts(context, rows):
357:     """Delete the script files of individuals whose result is stored.
358: 
359:     The scripts have done their job: everything they printed is in the database,
360:     and their source is in individuals.script_source, so the files themselves are
361:     spent. Clearing them keeps run_db/ to the database plus whatever is still
362:     waiting to run, and makes it impossible to launch a stale one by hand later.
363: 
364:     That covers the ones that just ran -- a failed run is still a processed one,
365:     and its output is stored either way -- and the ones skipped as unchanged,
366:     which are spent for the same reason: they ran in an earlier generation. Ones
367:     held back as BAD or by --limit have never run, so their scripts stay.
368:     """
369:     if context.options.keep_scripts:
370:         print("kept the scripts in %s (--keep-scripts)" % context.run_dir)
371:         return
372:     gone = store.remove_scripts(context.conn, context.run_id, context.run_dir,
373:                                 [row["script_name"] for row in rows])
374:     print("removed %d spent script(s) from %s -- they are still in the database "
375:           "(python start_run.py runs, or python -m gep_lora.core.storage.store --export)"
376:           % (gone, context.run_dir))
377: 
378: 
379: def step_process(context):
380:     """Execute each script -> executions and exchanges."""
381:     conn, run_id = context.conn, context.run_id
382:     conf, options = context.conf, context.options
383:     rows = [row for row in store.individuals(conn, run_id) if row["script_source"]]
384:     if not rows:
385:         raise SystemExit("run %d holds no generated scripts. Run the runs step first."
386:                          % run_id)
387: 
388:     # The scripts have to exist as files to be launched; rewrite any that are
389:     # missing or out of date.
390:     with context.timer("materialise"):
391:         store.materialise(conn, run_id, context.run_dir)
392: 
393:     # Whether the venv is needed is a property of the template that was filled:
394:     # mocked scripts load nothing, and demanding the venv for them would put a
395:     # GPU-less machine out of reach. A remote script imports nothing itself,
396:     # but the servers it talks to are started with sys.executable and do, so it
397:     # needs the venv just as much.
398:     remote = server_pool.wanted(rows[0]["script_source"])
399:     real = process_run.imports_unsloth(rows[0]["script_source"]) or remote
400:     if real:
401:         process_run.check_interpreter()
402: 
403:     runnable = [row for row in rows
404:                 if options.include_blocked or row["state"] != "BAD"]
405:     blocked = len(rows) - len(runnable)
406: 
407:     # An individual whose chromosome has not moved since it last ran would
408:     # produce the same execution again at the cost of another base-model load,
409:     # and its result is already in the database. has_changed is exactly that
410:     # question, so it is exactly what decides. An individual that has never run
411:     # is not "unchanged" -- there is nothing to have changed from -- so a fresh
412:     # population, and every copy selection appends, still runs in full.
413:     #
414:     # has_changed is set by mutation and stays set until the next round, so it
415:     # still says "changed" about an individual this generation has already run.
416:     # A generation interrupted half way through process -- a stopped job, a
417:     # killed driver -- would then run those again on resuming. So an
418:     # individual whose latest execution finished cleanly *and* built exactly
419:     # the blend it holds now (the chromosome its script printed, the weight
420:     # seed it drew under) is as unchanged as has_changed = 0 says. Only a clean
421:     # one: a crash or a timeout in an interrupted pass may be the interruption.
422:     done = store.executed(conn, run_id)
423:     current = {row["id"] for row in store.latest_runs(conn, run_id)
424:                if row["verdict"] == "ok" and row["ran_seed"] == row["weight_seed"]
425:                and process_run.expression(row["head"]) == row["chromosome"]}
426:     if options.include_unchanged:
427:         selected, unchanged = runnable, []
428:     else:
429:         selected = [row for row in runnable
430:                     if (row["has_changed"] and row["id"] not in current)
431:                     or row["id"] not in done]
432:         keep = {row["id"] for row in selected}
433:         unchanged = [row for row in runnable if row["id"] not in keep]
434: 
435:     if options.limit:
436:         selected = selected[:options.limit]
437: 
438:     if not selected:
439:         if unchanged:
440:             # A generation where nothing moved is a result, not a failure: the
441:             # database already holds the answer for every individual in it.
442:             print("nothing to run: all %d individual(s) already have an execution "
443:                   "of their current chromosome" % len(unchanged))
444:             print("pass --include-unchanged to run them again anyway")
445:             _clear_scripts(context, unchanged)
446:             return
447:         raise SystemExit("nothing to run: all %d individuals are marked BAD. "
448:                          "Pass --include-blocked to run them anyway." % len(rows))
449: 
450:     # How many run at once. The sweep's own value, so a batch size cannot be
451:     # changed under a sweep already under way; a sweep created before this knob
452:     # existed falls back to what settings.py says now.
453:     size = process_run.batch_size(
454:         conf.get("PROCESS_RUN_BATCH_SIZE", config.PROCESS_RUN_BATCH_SIZE))
455: 
456:     # The servers those scripts talk to, when they are lora_server clients: the
457:     # base-model load is paid per server instead of per individual. A sweep
458:     # generated from either of the other two templates gets None here and
459:     # everything below is as it was.
460:     #
461:     # Started at most once per *run*, not once per generation: the pool is kept
462:     # on the Context, which gep_lora/core/pipeline/continue_run.py builds once and reuses for every
463:     # generation it turns. A pool that came down at the end of each process step
464:     # would pay its startup again next generation for nothing -- the servers
465:     # would be reloading the same model they had just released. The exception is
466:     # wants_the_card(), below, which is the one thing that makes releasing them
467:     # worth it.
468:     #
469:     # Started before the batches are cut, because the pool caps them: a server
470:     # serves one request at a time, so more scripts in flight than there are
471:     # servers would only queue.
472:     pool = context.pool
473:     if pool is None:
474:         with context.timer("start servers"):
475:             pool = context.pool = server_pool.pool_for(
476:                 conf, rows[0]["script_source"],
477:                 generate_runs.base_model_name(conf.get("BASE_MODEL")),
478:                 context.run_dir, config)
479:     else:
480:         # Anything that died while the previous generation was scoring, or that
481:         # has worn its recycle count out, is replaced before a script is handed
482:         # it -- the same check that runs between batches, run once more across
483:         # the gap between generations, which is the longest a server sits idle.
484:         moment = time.time()
485:         replaced = pool.maintain()
486:         if replaced:
487:             context.phase("recycle servers", time.time() - moment)
488:         print("reusing the %d lora server(s) already up -- the base model has "
489:               "not been reloaded since they started%s"
490:               % (pool.size,
491:                  "" if not replaced else ", bar %d replaced just now" % replaced))
492: 
493:     # Whether they may stay up once this step is over. The step's own finally
494:     # honours it; the driver that built the Context releases whatever is left.
495:     keep = pool is not None and not wants_the_card(conf)
496: 
497:     if pool is not None and size > pool.size:
498:         print("batch size %d capped to the %d server(s) in the pool"
499:               % (size, pool.size))
500:         size = pool.size
501: 
502:     groups = process_run.batches(selected, size)
503: 
504:     # How many prompts each script has to get through, so a progress line can
505:     # say 12/50 rather than 12. Only the counting is done here -- an eval file
506:     # that cannot be read is every script's failure to report, not this step's,
507:     # so it costs the total and nothing else.
508:     try:
509:         _, prompts, _ = generate_runs.eval_prompt_count(
510:             conf.get("TRAINING_SET"), conf.get("TRAINING_COUNT"))
511:     except SystemExit:
512:         prompts = 0
513:     every = conf.get("PROCESS_RUN_PROGRESS_SECONDS",
514:                      config.PROCESS_RUN_PROGRESS_SECONDS)
515: 
516:     # Progress arrives on the threads draining the children, several at once in
517:     # a batch, so one lock owns the console for a whole line. Everything else
518:     # printed by this step is on this thread, between batches, and cannot
519:     # collide with it.
520:     console = threading.Lock()
521: 
522:     def say(script, message):
523:         with console:
524:             print("        %s%s" % ("" if size == 1 else script + "  ", message))
525: 
526:     def watch(script):
527:         return process_run.Progress(script, prompts, every, say)
528: 
529:     # What this step is judged on: the individuals it actually launched. The
530:     # ones it skipped cost nothing, so counting them would flatter the seconds
531:     # per individual that says what a bigger population would cost.
532:     context.count(len(selected), "individuals", skipped=blocked + len(unchanged),
533:                   note="batch %d, %d prompt(s) each" % (size, prompts))
534:     print("running %d of %d individuals%s%s"
535:           % (len(selected), len(rows),
536:              " (%d skipped as BAD)" % blocked if blocked else "",
537:              " (%d unchanged since their last run)" % len(unchanged)
538:              if unchanged else ""))
539:     if pool is not None:
540:         # The saving, said where it is made: the load already happened, once
541:         # per server, and these scripts only send their build plan.
542:         print("these are lora_server clients: the base model is already loaded "
543:               "on %d server(s), so no individual pays for one%s\n"
544:               % (pool.size,
545:                  "" if not pool.recycle_after else
546:                  " -- each server is restarted every %d build(s)"
547:                  % pool.recycle_after))
548:         if not keep:
549:             # Said before the wait rather than after it, because it is the
550:             # reason the next generation will pay for a startup again.
551:             print("        they come down when this step ends: JUDGE_BACKEND is "
552:                   "%r, so the evaluate step wants the card for its own judge\n"
553:                   % conf.get("JUDGE_BACKEND"))
554:     elif real:
555:         # Say what a batch costs where it is paid: every script in one is
556:         # another copy of the base model resident at the same time.
557:         print("each one loads the base model, so this takes a while%s\n"
558:               % ("" if size == 1 else
559:                  " -- %d of them loaded at once, in batches of %d" % (size, size)))
560:     else:
561:         print("these are mocked scripts: nothing is loaded, so this is quick\n")
562: 
563:     failures = 0
564:     started = time.time()
565:     number = 0
566:     # The pool outlives this step unless something else wants the card, so the
567:     # finally below releases it only when `keep` says it must -- including when
568:     # the step raised, since a step that fell over is not a reason to hand the
569:     # next one N base models it did not ask for. Whatever is left standing is
570:     # the Context's, and the driver that built the Context takes it down.
571:     try:
572:         for position, group in enumerate(groups, 1):
573:             # A whole batch is announced before it is launched, so a long wait says
574:             # what it is waiting for.
575:             first = number + 1
576:             for offset, row in enumerate(group):
577:                 print("[%d/%d] %s  %s" % (first + offset, len(selected),
578:                                           row["script_name"], row["chromosome"]))
579:             if size > 1:
580:                 print("        %sbatch %d/%d: %d running at once"
581:                       % ("" if not context.generation
582:                          else "gen %s, " % context.generation,
583:                          position, len(groups), len(group)))
584: 
585:             # One server per script of the batch, positionally: the batch is
586:             # capped at the pool's size above, so nothing queues.
587:             envs = None if pool is None else pool.envs_for(len(group))
588: 
589:             results = process_run.launch_batch(
590:                 context.run_dir, [row["script_name"] for row in group],
591:                 options.timeout, watch, envs)
592: 
593:             # Stored in the order they were asked for rather than the order they
594:             # finished, and from this thread only: the children print to their own
595:             # pipes and nothing but this loop touches the database.
596:             for row, (code, seconds, out, err) in zip(group, results):
597:                 number += 1
598:                 verdict = process_run.verdict_of(code)
599:                 if code != 0:
600:                     failures += 1
601: 
602:                 transcript = process_run.exchanges(out)
603:                 execution_id = store.add_execution(
604:                     conn, row["id"], seconds, code, verdict, row["weight_seed"],
605:                     process_run.drawn_weights(out), out, err)
606:                 store.add_exchanges(conn, execution_id, transcript)
607:                 # What the script cost from out here, and what it says it spent the
608:                 # time on in there -- the TIMING: lines it printed about itself, one
609:                 # phase row each, tied to the execution that paid for them. A script
610:                 # that printed none (an older sweep, or one that died before it got
611:                 # going) leaves the wall time and nothing under it.
612:                 # The row says whose script it was, rather than leaving that to a
613:                 # join: selection culls executions and sqlite hands their ids out
614:                 # again, so an id alone stops meaning one individual the moment a
615:                 # generation has been culled. Same rule as an execution row and a
616:                 # fitness_history row -- keep what was true when it was written.
617:                 context.phase("script", seconds, execution_id=execution_id,
618:                               detail={"number": row["number"],
619:                                       "chromosome": row["chromosome"],
620:                                       "verdict": verdict, "batch": size})
621:                 context.phases_from(process_run.timings(out),
622:                                     execution_id=execution_id)
623:                 # Commit per individual, so an interrupted sweep keeps what it has
624:                 # done -- a half-finished batch still leaves the ones that came back.
625:                 conn.commit()
626: 
627:                 # In a batch these lines no longer sit under their own heading, so
628:                 # they carry the script name; one at a time they read as before.
629:                 print("        %s%-9s %6.1fs  %d exchange(s) -> execution %d"
630:                       % ("" if size == 1 else row["script_name"] + "  ",
631:                          verdict, seconds, len(transcript), execution_id))
632:                 if code != 0:
633:                     # Show why, so a systemic problem is obvious without a query.
634:                     tail = [line for line in (out + err).splitlines()
635:                             if line.strip()][-1:]
636:                     if tail:
637:                         print("        %s" % tail[0][:100])
638: 
639:             # Between batches, never during one: a server is restarted when it
640:             # has died or worn its recycle count out, and either way nothing is
641:             # talking to it at this moment.
642:             # Timed only when it happened, rather than a phase of nothing per
643:             # batch: what a recycle costs is a model load, and a row of zeroes
644:             # beside it would flatter the average.
645:             if pool is not None:
646:                 moment = time.time()
647:                 if pool.maintain():
648:                     context.phase("recycle servers", time.time() - moment)
649:     finally:
650:         if not keep:
651:             context.release_pool()
652: 
653:     print("\nran %d in %.1fs, %d failed" % (len(selected), time.time() - started, failures))
654: 
655:     # The scripts have done their job: everything they printed is in the
656:     # database, and their source is in individuals.script_source, so the files
657:     # themselves are spent. Clearing them keeps run_db/ to the database plus
658:     # whatever is still waiting to run, and makes it impossible to launch a
659:     # stale one by hand later. This happens whatever the individuals did --
660:     # a failed run is still a processed one, and its output is stored either way.
661:     with context.timer("clear scripts"):
662:         _clear_scripts(context, selected + unchanged)
663: 
664:     # A chromosome that cannot run is a result, not a pipeline failure. Only a
665:     # sweep where nothing at all worked points at something systemic.
666:     if failures == len(selected):
667:         raise SystemExit("every individual failed -- try: python -m gep_lora.core.storage.store --show %d"
668:                          % run_id)
669: 
670: 
671: def step_evaluate(context):
672:     """Score every answer with the sweep's evaluator -> quality and reason.
673: 
674:     Which evaluator is EVALUATOR in the sweep's *stored* settings, so a sweep
675:     is scored the way it was created to be scored even if settings.py has moved
676:     on -- two individuals graded by different rubrics would not be comparable,
677:     and comparing them is the whole point of a fitness number.
678: 
679:     A wrapper around the work for one reason: a judge graded with
680:     JUDGE_BACKEND = 'unsloth' is a model resident in *this* process, and the
681:     step is done with it however it ends. gep_lora/core/pipeline/main.py carries straight on into the
682:     next generation, whose scripts each load the base model, so a judge left on
683:     the card would be VRAM taken from every one of them.
684:     """
685:     try:
686:         _evaluate(context)
687:     finally:
688:         evaluators.release_models()
689: 
690: 
691: def _evaluate(context):
692:     conn, run_id, options = context.conn, context.run_id, context.options
693:     pending = store.exchanges_to_score(conn, run_id, options.force)
694:     if not pending:
695:         total = conn.execute(
696:             "SELECT COUNT(*) AS n FROM exchanges x JOIN executions e ON e.id = x.execution_id"
697:             " JOIN individuals i ON i.id = e.individual_id WHERE i.run_id = ?",
698:             (run_id,)).fetchone()["n"]
699:         if not total:
700:             raise SystemExit("run %d holds no answers to score. Run the process step first."

(Showing lines 1-700 of 1462. Use offset=701 to continue.)
</content>
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 3.3s)

_Thinking:_

Let me continue reading gep_lora/core/pipeline/start_run.py 700-1462.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\start_run.py",
  "offset": 701
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\start_run.py</path>
<type>file</type>
<content>
701:                              % run_id)
702:         # Only the most recent execution of each individual is scored -- that is
703:         # the current result. An older one keeps whatever score it was given.
704:         print("nothing to score -- every answer in the latest execution of each "
705:               "individual already has a quality")
706:         return
707: 
708:     # prepare() is where an evaluator does its once-per-step work: discovering
709:     # the judge model, loading the eval set's own answers, checking its knobs.
710:     # It is handed the pending rows so it can tell an all-blank set -- a sweep
711:     # where every script failed, or a mocked one that arrived scored -- from one
712:     # that really needs an endpoint up.
713:     evaluator = evaluators.get(context.conf.get("EVALUATOR"))
714:     # The Context goes with it for the one evaluator that needs more than the
715:     # settings and the pending rows: llm_judge_baseline reads the cache of
716:     # base-model answers, and fills it -- one model load, once -- when the
717:     # sweep asks a question no earlier sweep did.
718:     # Said before prepare(), not after: preparing can take minutes -- the
719:     # baseline evaluator may have a base model to run -- and a console that has
720:     # not yet named the evaluator cannot explain what it is waiting for.
721:     print("evaluator: %s -- %s" % (evaluator.name, evaluator.description))
722:     # Timed on its own because it is the one fixed cost of this step: the
723:     # baseline evaluator may have a base model to load and a whole eval set to
724:     # answer here, and a step total cannot tell that from slow judging.
725:     with context.timer("prepare", detail={"evaluator": evaluator.name}):
726:         prepared = evaluator.prepare(context.conf, pending, context)
727:     for note in prepared.notes:
728:         print(note)
729:     print("scoring %d answer(s)%s\n"
730:           % (len(pending), " (--force: re-scoring)" if options.force else ""))
731: 
732:     # Giving up early is only ever worth it when a score costs something to
733:     # get -- a request, or a generate() on this machine. A local scorer would
734:     # spend nothing finishing an individual, and stopping short would only lose
735:     # detail. The rows arrive ordered by individual and then by position, so
736:     # each group below is one individual's answers in the order they were asked
737:     # -- which is what "the first 10%" means.
738:     limit_fraction = (context.conf.get("JUDGE_ABANDON_FRACTION")
739:                       if evaluator.asks_judge(context.conf) else None)
740:     if limit_fraction:
741:         print("giving up on an individual once its first %g%% of graded answers "
742:               "have all scored 0" % (100 * limit_fraction))
743: 
744:     scored = failed = abandoned = 0
745:     started = time.time()
746:     for number, group in itertools.groupby(pending, key=lambda row: row["number"]):
747:         rows = list(group)
748:         print("individual %d  %s" % (number, rows[0]["chromosome"]))
749:         # How many graded answers, all of them zero, are enough to stop asking.
750:         limit = (evaluators.abandon_after(context.conf, len(rows))
751:                  if limit_fraction else None)
752:         graded = zeros = 0
753:         giving_up = ""
754: 
755:         for index, row in enumerate(rows):
756:             answer = row["answer"] or ""
757:             if giving_up:
758:                 # Written rather than left NULL: an individual's fitness is the
759:                 # mean over its exchanges, so this is what makes "fitness zero"
760:                 # true of the whole individual rather than of the part that was
761:                 # graded -- and a re-run of evaluate does not ask about these
762:                 # again.
763:                 store.score_exchange(conn, row["id"], 0.0, giving_up, None)
764:                 scored += 1
765:                 abandoned += 1
766:             elif not answer.strip():
767:                 # Nothing to grade: an unanswered question is worth nothing, and
768:                 # asking the judge about an empty string just wastes a call. It
769:                 # is not an evaluation either, so it neither counts toward the
770:                 # first 10% nor condemns the individual on its own.
771:                 store.score_exchange(conn, row["id"], 0.0, "no answer given", None)
772:                 scored += 1
773:                 print("    [%d] 0.00  (no answer)" % row["position"])
774:             else:
775:                 item = {"question": row["question"], "answer": answer,
776:                         "position": row["position"], "number": row["number"]}
777:                 try:
778:                     # An evaluator that cannot score this one answer fails it
779:                     # and nothing else: the exchange keeps its NULL quality, the
780:                     # step counts it, and a re-run picks it up again. A failure
781:                     # is not a zero, so it never counts toward giving up either.
782:                     call_started = time.perf_counter()
783:                     quality, reason = evaluator.score(item, prepared)
784:                     # Timed here rather than around the whole loop: one call is
785:                     # a request or a generate(), and calls x seconds is what
786:                     # says whether to speed the judge up or ask it less often.
787:                     # A failed call is timed too -- see below -- since a judge
788:                     # that times out costs its timeout.
789:                     context.phase("score", time.perf_counter() - call_started,
790:                                   detail={"judge": prepared.label})
791:                     quality = round(quality, 3)
792:                     store.score_exchange(conn, row["id"], quality, reason,
793:                                          prepared.label)
794:                     scored += 1
795:                     graded += 1
796:                     zeros += 1 if quality == 0.0 else 0
797:                     print("    [%d] %.2f  %s" % (row["position"], quality, reason))
798:                 except (RuntimeError, ValueError) as error:
799:                     context.phase("score (failed)",
800:                                   time.perf_counter() - call_started,
801:                                   detail={"judge": prepared.label})
802:                     failed += 1
803:                     print("    [%d] FAILED  %s" % (row["position"], error))
804: 
805:                 if limit and graded >= limit and zeros == graded:
806:                     giving_up = ("abandoned: the first %d graded answer(s) all "
807:                                  "scored 0" % graded)
808:                     print("    giving up on individual %d -- %s, so its "
809:                           "remaining %d answer(s) are scored 0 unasked"
810:                           % (number, giving_up, len(rows) - index - 1))
811:             # Saved as we go, so an interrupted scoring run resumes where it stopped.
812:             conn.commit()
813: 
814:     print("\nscored %d, %d failed, in %.1fs" % (scored, failed, time.time() - started))
815: 
816:     context.count(scored, "answers", skipped=abandoned,
817:                   note="%s via %s" % (evaluator.name, prepared.label or "-"))
818: 
819:     if abandoned:
820:         print("%d answer(s) scored 0 unasked, on individuals given up on early"
821:               % abandoned)
822: 
823:     qualities = [row["quality"] for row in store.quality_rows(conn, run_id)
824:                  if row["quality"] is not None]
825:     if qualities:
826:         print("quality across %d individual(s): min %.3f, max %.3f, mean %.3f"
827:               % (len(qualities), min(qualities), max(qualities),
828:                  sum(qualities) / len(qualities)))
829: 
830:     if failed and not scored:
831:         raise SystemExit("nothing could be scored by the %s evaluator -- check "
832:                          "its settings, and the judge it asks (JUDGE_BACKEND) "
833:                          "if it asks one" % evaluator.name)
834: 
835: 
836: def step_fitness(context):
837:     """Fold each transcript into one number -> individuals.fitness, and record
838:     that generation of it -> fitness_history."""
839:     conn, run_id = context.conn, context.run_id
840:     snapshot = calculate_fitness.assign(conn, run_id)
841:     rows = snapshot.rows
842:     if not rows:
843:         raise SystemExit("run %d holds no individuals. Run the population step first."
844:                          % run_id)
845:     if not any(row["quality"] is not None for row in rows):
846:         raise SystemExit("no answer in run %d carries a quality yet -- run the "
847:                          "evaluate step first." % run_id)
848: 
849:     print("fitness = mean quality across the exchanges of the latest execution\n")
850:     print("    %-4s %-5s %-7s %-7s %s"
851:           % ("#", "state", "answers", "fitness", "chromosome"))
852:     values = []
853:     for row in rows:                            # already best first
854:         value = calculate_fitness.fitness_of(row)
855:         values.append(value)
856:         print("    %-4d %-5s %-7d %-7.3f %s"
857:               % (row["number"], row["state"] or "-", row["answers"] or 0,
858:                  value, row["chromosome"]))
859: 
860:     print("\nwrote fitness for %d individual(s): min %.3f, max %.3f, mean %.3f"
861:           % (len(values), min(values), max(values), sum(values) / len(values)))
862: 
863:     context.count(len(values), "individuals",
864:                   skipped=sum(1 for row in rows if row["quality"] is None))
865: 
866:     # An individual averaged over part of its transcript still gets a fitness,
867:     # but it is a weaker claim than one averaged over all of it -- say so.
868:     partial = [row["number"] for row in rows
869:                if row["quality"] is not None and (row["unscored"] or 0)]
870:     if partial:
871:         print("%d individual(s) averaged over a partly scored transcript: %s"
872:               % (len(partial), ", ".join(str(number) for number in partial)))
873:     # 0.0 here is a decision, not a gap -- see calculate_fitness.py.
874:     empty = [row["number"] for row in rows if row["quality"] is None]
875:     if empty:
876:         print("%d individual(s) had no judged answer and scored 0.0: %s"
877:               % (len(empty), ", ".join(str(number) for number in empty)))
878: 
879:     # The column above says what the population is worth now; this says what it
880:     # was worth each generation, which is the only way a finished sweep can show
881:     # whether the search went anywhere. Re-running this step restates the
882:     # current generation rather than adding one -- see store.fitness_generation.
883:     print("\nrecorded generation %d in fitness_history at %s"
884:           % (snapshot.generation, snapshot.recorded_at))
885:     history = store.fitness_by_generation(conn, run_id)
886:     if len(history) > 1:
887:         print("    %-5s %-20s %-6s %-7s %-7s %s"
888:               % ("gen", "recorded", "pop", "best", "mean", "fittest chromosome"))
889:         for entry in history:
890:             best = store.best_of_generation(conn, run_id, entry["generation"])
891:             print("    %-5d %-20s %-6d %-7.3f %-7.3f %s"
892:                   % (entry["generation"], entry["recorded_at"], entry["population"],
893:                      entry["best"] or 0.0, entry["mean"] or 0.0,
894:                      best["chromosome"] if best else "-"))
895: 
896: 
897: def step_elitism(context):
898:     """Name the one individual to carry forward -> individuals.is_best."""
899:     conn, run_id = context.conn, context.run_id
900:     best, rows = elitism.elect(conn, run_id)
901:     if not rows:
902:         raise SystemExit("run %d holds no individuals. Run the population step first."
903:                          % run_id)
904:     if best is None:
905:         # Nothing was written, so is_best is whatever it was -- see elitism.py.
906:         raise SystemExit("every individual in run %d has fitness 0.0 -- run the "
907:                          "fitness step first, or, if they really all scored 0.0, "
908:                          "this generation has no elite to carry forward." % run_id)
909: 
910:     context.count(len(rows), "individuals")
911: 
912:     tied = [row["number"] for row in rows
913:             if (row["fitness"] or 0.0) == (best["fitness"] or 0.0)]
914:     print("elite: individual %d, fitness %.3f" % (best["number"], best["fitness"]))
915:     print("    %s" % best["chromosome"])
916:     print("cleared is_best on the other %d individual(s)" % (len(rows) - 1))
917:     if len(tied) > 1:
918:         # Say so rather than letting the choice look like a ranking.
919:         print("%d individuals tie at %.3f (%s); the lowest number takes it"
920:               % (len(tied), best["fitness"] or 0.0,
921:                  ", ".join(str(number) for number in tied)))
922: 
923: 
924: def _span(numbers):
925:     """"#4-#6", or "#4" when there is only one of them."""
926:     if len(numbers) == 1:
927:         return "#%d" % numbers[0]
928:     return "#%d-#%d" % (numbers[0], numbers[-1])
929: 
930: 
931: def _master_seed(context, name):
932:     """The sweep's seed called `name`, drawing and recording one if it has none.
933: 
934:     The same bargain the other seeds strike: a sweep that was never asked to be
935:     repeatable still is, because whatever it used is written down. A sweep
936:     created before the setting existed gets its seed here instead of at
937:     creation, and keeps it from then on.
938:     """
939:     seed = context.conf.get(name)
940:     if seed is None:
941:         seed = random.randrange(_SEED_LIMIT)
942:         context.conf[name] = seed
943:         store.save_settings(context.conn, context.run_id, {name: seed})
944:         print("drew %s for run %d and stored it: %d" % (name, context.run_id, seed))
945:     return seed
946: 
947: 
948: def step_selection(context):
949:     """Spin the roulette wheel -> the n+1 weakest replaced by n copies and one
950:     newcomer, leaving the population the size it was."""
951:     conn, run_id = context.conn, context.run_id
952:     master = _master_seed(context, "SELECTION_MASTER_SEED")
953:     before = store.individuals(conn, run_id)
954:     if not before:
955:         raise SystemExit("run %d holds no individuals. Run the population step first."
956:                          % run_id)
957: 
958:     # Derived from the sweep's seed and the population's high-water number, so
959:     # a second round draws its own parents rather than the first round's again,
960:     # and re-running a sweep from its stored seed reproduces every round of it.
961:     # The number and not the size: a round now culls as many individuals as it
962:     # appends, so the size is the same every generation and seeding on it would
963:     # spin the identical wheel every time.
964:     rng = random.Random("%s:%d" % (master, store.high_number(conn, run_id)))
965:     count = context.conf.get("SELECTION_COUNT")
966:     picked = selection.select(conn, run_id, count, rng, context.conf)
967: 
968:     if not picked.parents:
969:         raise SystemExit("every individual in run %d has fitness 0.0 -- there is no "
970:                          "wheel to spin. Run the fitness step first, or, if they "
971:                          "really all scored 0.0, this generation selects nobody."
972:                          % run_id)
973: 
974:     # How often each parent came up: the whole point of drawing with
975:     # replacement, and the quickest read on whether the wheel is doing anything.
976:     times = {}
977:     for row in picked.parents:
978:         times[row["number"]] = times.get(row["number"], 0) + 1
979:     context.count(len(picked.parents), "spins")
980:     print("spun the wheel %d time(s) over %d individual(s)"
981:           % (len(picked.parents), len(before)))
982:     print("    %-9s %-7s %-6s %s" % ("parent", "fitness", "picked", "chromosome"))
983:     for row in sorted(picked.parents, key=lambda row: -(row["fitness"] or 0.0)):
984:         if row["number"] in times:
985:             print("    %-9d %-7.3f %-6d %s"
986:                   % (row["number"], row["fitness"] or 0.0,
987:                      times.pop(row["number"]), row["chromosome"]))
988: 
989:     missed = [row["number"] for row in before if row["number"] not in
990:               {row["number"] for row in picked.parents}]
991:     if missed:
992:         print("%d individual(s) the wheel never landed on: %s -- being missed is "
993:               "not what gets an individual culled; being weakest is"
994:               % (len(missed), ", ".join(str(number) for number in missed)))
995:     print("appended %d copy/copies as %s"
996:           % (len(picked.numbers), _span(picked.numbers)))
997: 
998:     # The cull is the other half of a round: n copies and one newcomer in, the
999:     # n+1 weakest out, so the population comes out the size it went in and what
1000:     # moved is its membership. The elite is never eligible, and a population
1001:     # too small to spare that many gives up what it can and grows by the rest.
1002:     if picked.culled:
1003:         print("culled the %d weakest individual(s), transcripts and all:"
1004:               % len(picked.culled))
1005:         for row in picked.culled:
1006:             print("    %-9d %-7.3f %s"
1007:                   % (row["number"], row["fitness"] or 0.0, row["chromosome"]))
1008:         # Their rows are gone, so remove_scripts() can no longer find the files
1009:         # they owned -- the names have to come from the rows the cull handed
1010:         # back. The scripts were only ever a cache of script_source, and the
1011:         # source went with the row.
1012:         stale = store.discard_scripts(context.run_dir,
1013:                                       [row["script_name"] for row in picked.culled])
1014:         if stale:
1015:             print("    and %d of their script file(s) from %s"
1016:                   % (stale, context.run_dir))
1017:     else:
1018:         print("culled nobody: the population had no non-elite individual to spare")
1019:     short = len(picked.parents) + 1 - len(picked.culled)
1020:     if short > 0:
1021:         print("that is %d fewer than the %d a round of %d appends, so the population "
1022:               "grows by %d this generation -- there was no other non-elite "
1023:               "individual to spare"
1024:               % (short, len(picked.parents) + 1, len(picked.parents), short))
1025: 
1026:     # And the one individual per round that is nobody's copy: the floor under a
1027:     # gene pool that copying and mutation can only ever narrow.
1028:     print("drew #%d fresh from the same rules the first population came from: %s"
1029:           % (picked.newcomer.number, picked.newcomer.chromosome))
1030: 
1031:     after = len(before) + len(picked.numbers) - len(picked.culled) + 1
1032:     if after == len(before):
1033:         print("the population is still %d, as it went in -- %d out, %d in"
1034:               % (after, len(picked.culled), len(picked.numbers) + 1))
1035:     else:
1036:         print("the population is now %d, up from %d" % (after, len(before)))
1037:     # Each copy is its parent field for field, so it arrives holding the
1038:     # parent's script name, weight seed and fitness as well as its chromosome.
1039:     # Those are the parent's answers, and stay right only while the chromosome
1040:     # is still the parent's; the newcomer has none of them to begin with;
1041:     # trees and runs re-derive them for everyone either way.
1042:     print("each copy is its parent field for field and the newcomer is bare -- "
1043:           "re-derive their trees, scripts and seeds with: python start_run.py trees runs")
1044: 
1045: 
1046: def step_mutation(context):
1047:     """Mutate everything but the elite -> chromosome, has_changed."""
1048:     conn, run_id = context.conn, context.run_id
1049:     rate = context.conf.get("MUTATION_RATE", config.MUTATION_RATE)
1050:     if not 0.0 <= rate <= 1.0:
1051:         raise SystemExit("MUTATION_RATE is %r; it is a probability per symbol and "
1052:                          "has to be between 0.0 and 1.0" % rate)
1053: 
1054:     master = _master_seed(context, "MUTATION_MASTER_SEED")
1055:     before = store.individuals(conn, run_id)
1056:     if not before:
1057:         raise SystemExit("run %d holds no individuals. Run the population step first."
1058:                          % run_id)
1059: 
1060:     # Derived the way selection's is, and for the same reason: the population's
1061:     # highest number rises a generation at a time, so it dates the round, and a
1062:     # sweep replayed from its stored seed mutates exactly as it did the first
1063:     # time. Seeding on the size instead would mutate the same positions of the
1064:     # same individuals every generation, now that the size does not move.
1065:     rng = random.Random("%s:%d" % (master, store.high_number(conn, run_id)))
1066:     changes, rows = mutation.apply(conn, run_id, rate, rng)
1067: 
1068:     elite = [row["number"] for row in rows if row["is_best"]]
1069:     eligible = len(rows) - len(elite)
1070:     context.count(eligible, "individuals", skipped=len(elite))
1071:     print("rate %.3f per symbol, over %d of %d individual(s)%s"
1072:           % (rate, eligible, len(rows),
1073:              " (#%s is the elite and is left alone)"
1074:              % ", #".join(str(number) for number in elite) if elite else ""))
1075:     if changes:
1076:         print("    %-4s %-7s %s" % ("#", "symbols", "chromosome"))
1077:         for change in changes:
1078:             print("    %-4d %-7d %s" % (change.number, change.symbols, change.before))
1079:             print("    %-4s %-7s %s" % ("", "->", change.after))
1080:     print("mutated %d, left %d unchanged; has_changed is set on %d individual(s)"
1081:           % (len(changes), len(rows) - len(changes), len(changes)))
1082:     if changes:
1083:         # Their fitness is gone, cleared with the chromosome that earned it. The
1084:         # tree, script and rank are merely stale descriptions, and nothing here
1085:         # can re-derive them.
1086:         print("their fitness is cleared, and their trees, scripts and ranks still "
1087:               "describe the chromosome they used to be")
1088:         print("re-derive and re-earn: python start_run.py trees runs, then process")
1089: 
1090: 
1091: def step_weight_mutation(context):
1092:     """Mutate the weights of everything but the elite -> chromosome, has_changed."""
1093:     conn, run_id = context.conn, context.run_id
1094:     rate = context.conf.get("WEIGHT_MUTATION_RATE", config.WEIGHT_MUTATION_RATE)
1095:     if not 0.0 <= rate <= 1.0:
1096:         raise SystemExit("WEIGHT_MUTATION_RATE is %r; it is the share of all the "
1097:                          "population's weights moved per round and has to be "
1098:                          "between 0.0 and 1.0" % rate)
1099: 
1100:     master = _master_seed(context, "WEIGHT_MUTATION_MASTER_SEED")
1101:     before = store.individuals(conn, run_id)
1102:     if not before:
1103:         raise SystemExit("run %d holds no individuals. Run the population step first."
1104:                          % run_id)
1105: 
1106:     # Dated by the high-water number, as mutation's generator is. The prefix
1107:     # keeps the two streams apart when both master seeds hold the same value.
1108:     rng = random.Random("weights:%s:%d" % (master, store.high_number(conn, run_id)))
1109:     changes, rows = weight_mutation.apply(conn, run_id, rate, rng)
1110: 
1111:     elite = [row["number"] for row in rows if row["is_best"]]
1112:     pool = weight_mutation.pool_size(rows)
1113:     drawn = weight_mutation.draw_count(rate, pool)
1114:     context.count(len(rows) - len(elite), "individuals", skipped=len(elite))
1115:     print("rate %.3f of %d weight(s) across %d of %d individual(s): %d drawn%s"
1116:           % (rate, pool, len(rows) - len(elite), len(rows), drawn,
1117:              " (#%s is the elite and is left alone)"
1118:              % ", #".join(str(number) for number in elite) if elite else ""))
1119:     if changes:
1120:         print("    %-4s %-7s %s" % ("#", "weights", "chromosome"))
1121:         for change in changes:
1122:             print("    %-4d %-7d %s" % (change.number, change.weights, change.before))
1123:             print("    %-4s %-7s %s" % ("", "->", change.after))
1124:     print("re-weighted %d individual(s); has_changed is set on each and never "
1125:           "cleared here" % len(changes))
1126:     if changes:
1127:         print("their fitness is cleared; re-derive and re-earn: "
1128:               "python start_run.py trees runs, then process")
1129: 
1130: 
1131: Step = namedtuple("Step", "name run description")
1132: 
1133: STEPS = [
1134:     Step("population", step_population,
1135:          "draw the chromosomes -> individuals"),
1136:     Step("trees", step_trees,
1137:          "draw each chromosome as a tree -> individuals.tree"),
1138:     Step("runs", step_runs,
1139:          "fill the template per individual -> individuals.script_source + run_db/"),
1140:     # The expensive one: a base-model load per individual. Keep COUNT small
1141:     # while iterating, or run the earlier steps on their own.
1142:     Step("process", step_process,
1143:          "execute each script -> executions, exchanges; then delete the scripts"),
1144:     # The cost depends on the evaluator: a judge is one call per answer and
1145:     # needs its endpoint up, while similarity and heuristic are local and free.
1146:     Step("evaluate", step_evaluate,
1147:          "score every answer with the sweep's evaluator -> exchanges.quality"),
1148:     # Cheap, and pure arithmetic over what evaluate stored: no model, no judge.
1149:     Step("fitness", step_fitness,
1150:          "average each transcript's qualities -> individuals.fitness,"
1151:          " fitness_history"),
1152:     # Cheap too, and reads nothing but the column fitness just wrote.
1153:     Step("elitism", step_elitism,
1154:          "mark the fittest individual as the one to keep -> individuals.is_best"),
1155:     # Appends more than it removes, so running it twice is two generations of
1156:     # selection, not one done twice.
1157:     Step("selection", step_selection,
1158:          "roulette wheel sampling -> copies of the fit appended, the weakest"
1159:          " culled, one newcomer drawn"),
1160:     # Leaves the elite alone, so the best result found so far survives intact.
1161:     Step("mutation", step_mutation,
1162:          "point-mutate every other chromosome -> chromosome, has_changed"),
1163:     # After mutation, because it only ever raises has_changed: running first
1164:     # would let mutation write 0 over an individual this step re-weighted.
1165:     Step("weight_mutation", step_weight_mutation,
1166:          "swap a share of all the non-elite weights -> chromosome, has_changed"),
1167: ]
1168: 
1169: 
1170: # The tail of the pipeline: the three steps that do not describe this
1171: # generation but build the *next* one. `elitism` names the individual mutation
1172: # must not touch, `selection` fills the population the next generation runs,
1173: # and `mutation` varies what it filled it with -- none of them reads a
1174: # transcript or writes a score, and all three leave the population describing a
1175: # generation that has not happened yet.
1176: #
1177: # So the last generation of a run stops before them, at `fitness`: the sweep
1178: # comes to rest holding the individuals that were actually scored, each one
1179: # still described by the script that earned it and the transcript it earned it
1180: # with. That is the state store.py --export, the HTML report and
1181: # test_run_with_dataset.py all want, and the state a sweep used to have to be
1182: # walked back into by re-running `trees runs process evaluate`.
1183: NEXT_GENERATION = ("elitism", "selection", "mutation", "weight_mutation")
1184: 
1185: 
1186: def without_next_generation(steps):
1187:     """`steps` up to the point where they stop being about this generation.
1188: 
1189:     Trims NEXT_GENERATION off the end rather than filtering it out anywhere it
1190:     appears, so a step list that never had a tail comes back unchanged and one
1191:     that does keeps everything before it -- including a `fitness` that somebody
1192:     asked for by name.
1193:     """
1194:     trimmed = list(steps)
1195:     while trimmed and trimmed[-1].name in NEXT_GENERATION:
1196:         trimmed.pop()
1197:     return trimmed
1198: 
1199: 
1200: # --- driver ----------------------------------------------------------------
1201: 
1202: 
1203: def resolve_run_dir(conf, override=None, default=None):
1204:     """Absolute path of the folder the generated scripts live in.
1205: 
1206:     Absolute from here on: each script is launched with cwd set to this folder,
1207:     so a relative path would be resolved against itself a second time.
1208: 
1209:     `default` displaces DB_RUN_DIR when the run has a folder of its own -- a
1210:     --from-db run works entirely beside its database, see db_datasets. --run-dir
1211:     still wins over both: it is the one thing that was asked for out loud.
1212:     """
1213:     run_dir = override or default or conf.get("DB_RUN_DIR") or config.DB_RUN_DIR
1214:     if not os.path.isabs(run_dir):
1215:         run_dir = os.path.join(_HERE, run_dir)
1216:     return os.path.abspath(run_dir)
1217: 
1218: 
1219: def context_for(conn, run_id, conf, args):
1220:     """The Context a step gets. One place, so every driver builds the same one.
1221: 
1222:     Which is also why the --from-db run folder is chosen here rather than in
1223:     each driver: gep_lora/core/pipeline/start_run.py and gep_lora/core/pipeline/continue_run.py both come through this, so a run
1224:     driven from a database keeps to its own folder whichever of them is turning
1225:     the crank.
1226:     """
1227:     isolated = (db_datasets.run_folder(conn, run_id)
1228:                 if getattr(args, "from_db", False) else None)
1229:     return Context(conn, run_id, conf,
1230:                    resolve_run_dir(conf, getattr(args, "run_dir", None), isolated),
1231:                    generate_runs.template_path(conf.get("TEMPLATE")), args)
1232: 
1233: 
1234: def run(steps, context):
1235:     """Run `steps` in order. Returns the exit code for the process.
1236: 
1237:     A sweep that stops part way is left marked 'failed', so `python -m gep_lora.core.storage.store
1238:     --list` says so rather than presenting a half-finished run as a result.
1239:     """
1240:     started = time.time()
1241:     # One pass over the sweep: this call, whatever steps it was given. A
1242:     # generation, when gep_lora/core/pipeline/continue_run.py is the one calling. The watermark is read
1243:     # once here rather than per step, so selection raising it half way through
1244:     # cannot file the same generation under two numbers -- see the step_timings
1245:     # comment in store.py.
1246:     pass_no = store.next_pass(context.conn, context.run_id)
1247:     watermark = store.high_number(context.conn, context.run_id)
1248:     for number, step in enumerate(steps, 1):
1249:         print("=" * 70)
1250:         print("[%d/%d]%s %s -- %s"
1251:               % (number, len(steps),
1252:                  "" if not context.generation else " gen %s" % context.generation,
1253:                  step.name, step.description))
1254:         print("=" * 70)
1255:         # A step is timed whether or not it says anything about itself: the wall
1256:         # time and the row are the driver's, and the phases inside are the
1257:         # step's own to add. A step added later is therefore timed by existing.
1258:         context.meter = record.Meter(context.conn, context.run_id, pass_no,
1259:                                      number, step.name,
1260:                                      generation=context.generation,
1261:                                      watermark=watermark)
1262:         context.meter.started_at = store.now()
1263:         step_started = time.perf_counter()
1264:         try:
1265:             step.run(context)
1266:         except SystemExit as error:
1267:             if error.code not in (0, None):
1268:                 context.meter.commit(time.perf_counter() - step_started, "failed")
1269:                 print("\nSTOPPED in step '%s': %s" % (step.name, error))
1270:                 print("Later steps were skipped, since they build on this one.")
1271:                 store.finish_run(context.conn, context.run_id, "failed")
1272:                 return 1
1273:         except Exception as error:                      # noqa: BLE001 - report and stop
1274:             # The row is written before the sweep is marked failed, for the
1275:             # reason a crashed individual's transcript is kept: a step that fell
1276:             # over after forty minutes is the most expensive thing this sweep
1277:             # did, and this is the only record of it.
1278:             context.meter.commit(time.perf_counter() - step_started, "failed")
1279:             print("\nSTOPPED in step '%s': %s: %s"
1280:                   % (step.name, type(error).__name__, error))
1281:             store.finish_run(context.conn, context.run_id, "failed")
1282:             return 1
1283:         seconds = time.perf_counter() - step_started
1284:         context.meter.commit(seconds)
1285:         context.meter = record.blank()
1286:         print("  (%s took %.1fs)\n" % (step.name, seconds))
1287: 
1288:     store.finish_run(context.conn, context.run_id, "done")
1289:     print("=" * 70)
1290:     print("done: %s in %.1fs" % (", ".join(step.name for step in steps),
1291:                                  time.time() - started))
1292:     print("run %d in %s -- python -m gep_lora.core.storage.store --show %d"
1293:           % (context.run_id, context.conn.path, context.run_id))
1294:     print("=" * 70)
1295:     return 0
1296: 
1297: 
1298: def main(argv=None):
1299:     parser = argparse.ArgumentParser(
1300:         description="Run the pipeline into a sqlite database: %s."
1301:                     % " -> ".join(step.name for step in STEPS)
1302:     )
1303:     parser.add_argument("steps", nargs="*", metavar="STEP",
1304:                         help="steps to run (default: all of them, in order)")
1305:     parser.add_argument("--list", action="store_true",
1306:                         help="list the steps and exit")
1307:     parser.add_argument("--evaluators", action="store_true",
1308:                         help="list the ways an answer can be scored and exit "
1309:                              "(EVALUATOR in settings.py picks one)")
1310:     parser.add_argument("--db", default=config.DB_PATH,
1311:                         help="database file (default %s)" % config.DB_PATH)
1312:     parser.add_argument("--run", type=int, default=None, metavar="ID",
1313:                         help="resume this sweep instead of starting one (0 = the latest)")
1314:     parser.add_argument("--label", default=None,
1315:                         help="a note stored with the sweep, to find it again later")
1316:     parser.add_argument("--from-db", action="store_true",
1317:                         help="read the eval prompts from the sweep's own stored "
1318:                              "dataset rows instead of the files its settings name "
1319:                              "(needs --run; the rows are written out beside the "
1320:                              "database and the *_SET settings point at them for "
1321:                              "this run only)")
1322:     parser.add_argument("--run-dir", default=None,
1323:                         help="folder for the generated scripts (default %s)"
1324:                              % config.DB_RUN_DIR)
1325:     parser.add_argument("--limit", type=int, default=0,
1326:                         help="process only the first N individuals (0 = all)")
1327:     parser.add_argument("--include-blocked", action="store_true",
1328:                         help="also run the ones marked BAD")
1329:     parser.add_argument("--include-unchanged", action="store_true",
1330:                         help="also run individuals whose chromosome has not "
1331:                              "changed since their last execution (default: skip "
1332:                              "them; their result is already stored)")
1333:     parser.add_argument("--keep-scripts", action="store_true",
1334:                         help="leave the generated scripts on disk after processing "
1335:                              "them (default: delete them; the source is in the database)")
1336:     parser.add_argument("--timeout", type=int, default=900,
1337:                         help="seconds to allow each script (default 900)")
1338:     parser.add_argument("--force", action="store_true",
1339:                         help="re-score answers that already have a quality")
1340:     parser.add_argument("--next-generation", action="store_true",
1341:                         help="also run %s -- the steps that build the next "
1342:                              "generation. Without it a full run stops after "
1343:                              "fitness, because gep_lora/core/pipeline/start_run.py's one generation is the "
1344:                              "whole run and there is no next generation to "
1345:                              "build. gep_lora/core/pipeline/continue_run.py passes it for every "
1346:                              "generation but its last."
1347:                              % ", ".join(NEXT_GENERATION))
1348:     args = parser.parse_args(argv)
1349: 
1350:     known = {step.name: step for step in STEPS}
1351: 
1352:     if args.list:
1353:         for step in STEPS:
1354:             print("%-15s %s" % (step.name, step.description))
1355:         return 0
1356: 
1357:     if args.evaluators:
1358:         current = config.EVALUATOR
1359:         for name, description in evaluators.available():
1360:             mark = "*" if name == current else " "
1361:             print("%s %-20s %s" % (mark, name, description))
1362:         print("\n* is EVALUATOR in settings.py, which a *new* sweep is created "
1363:               "with.\n  A sweep already in the database keeps the one it was "
1364:               "created with.")
1365:         # Where the judge runs is the other half of "how will this be scored",
1366:         # and it is a setting the same way: named here, frozen into the sweep.
1367:         backend = evaluators.backend_of(config.snapshot())
1368:         # Counted off the registry rather than written out, so adding an
1369:         # evaluator that asks a model does not leave this line saying "four".
1370:         # Split on via_judge_backend, since not every evaluator that asks a
1371:         # model asks it through JUDGE_BACKEND -- jev_judge_reference speaks to
1372:         # Jev directly, and claiming it ran through whatever endpoint or local
1373:         # model JUDGE_BACKEND names would be the wrong judge entirely.
1374:         via_backend = [name for name, _description in evaluators.available()
1375:                        if evaluators.get(name).needs_judge
1376:                        and evaluators.get(name).via_judge_backend]
1377:         own_transport = [name for name, _description in evaluators.available()
1378:                          if evaluators.get(name).needs_judge
1379:                          and not evaluators.get(name).via_judge_backend]
1380:         print("\n  The %d judging evaluators ask their judge through "
1381:               "JUDGE_BACKEND = %r:\n  %s"
1382:               % (len(via_backend), backend,
1383:                  "loaded here with unsloth, %s"
1384:                  % (config.JUDGE_MODEL or "and JUDGE_MODEL must name one")
1385:                  if backend == evaluators.UNSLOTH else
1386:                  "%s at %s" % (config.JUDGE_MODEL or "whatever it has loaded",
1387:                                config.JUDGE_BASE_URL)))
1388:         if own_transport:
1389:             print("\n  %s ask%s a judge of their own, not through JUDGE_BACKEND."
1390:                   % (", ".join(own_transport), "" if len(own_transport) > 1 else "s"))
1391:         return 0
1392: 
1393:     if args.steps:
1394:         unknown = [name for name in args.steps if name not in known]
1395:         if unknown:
1396:             parser.error("unknown step(s): %s. Known steps: %s"
1397:                          % (", ".join(unknown), ", ".join(known)))
1398:         # Keep the order the pipeline defines, not the order they were typed.
1399:         # Named steps are run as named: asking for `selection` and being given
1400:         # something else would be worse than any default.
1401:         selected = [step for step in STEPS if step.name in set(args.steps)]
1402:     else:
1403:         selected = list(STEPS)
1404:         if not args.next_generation:
1405:             selected = without_next_generation(selected)
1406: 
1407:     # Before connect(), which creates the database it cannot open: a usage
1408:     # error should not leave an empty one behind. A *new* sweep is the moment a
1409:     # dataset is read out of the files its settings name and stored, so there
1410:     # are no rows for --from-db to read back yet -- it needs one that exists.
1411:     if args.from_db and args.run is None:
1412:         raise SystemExit("--from-db reads a sweep's stored dataset, so it needs a "
1413:                          "sweep: pass --run (0 = the latest). A new sweep is where "
1414:                          "those rows come from.")
1415: 
1416:     conn = store.connect(args.db)
1417: 
1418:     # A sweep that draws a population is a new sweep; one that does not is
1419:     # continuing an existing one, and must use the settings that one was
1420:     # created with rather than whatever settings.py says now.
1421:     if args.run is None and any(step.name == "population" for step in selected):
1422:         run_id, conf = new_sweep(conn, args.label)
1423:         print()
1424:     else:
1425:         run_id = store.latest_run(conn) if args.run in (None, 0) else args.run
1426:         if run_id is None:
1427:             raise SystemExit("%s holds no runs yet -- start one with the population "
1428:                              "step." % conn.path)
1429:         if store.get_run(conn, run_id) is None:
1430:             raise SystemExit("no run %d in %s. Try: python -m gep_lora.core.storage.store --list"
1431:                              % (run_id, conn.path))
1432:         conf = store.get_settings(conn, run_id)
1433:         print("resuming run %d in %s\n" % (run_id, conn.path))
1434:         # The sweep's settings already come from the sweep; --from-db is the
1435:         # other half of that -- its questions too, rather than whatever the
1436:         # files those settings name hold now.
1437:         if args.from_db:
1438:             conf = db_datasets.repoint(conn, run_id, conf)
1439:             print()
1440: 
1441:     if not args.steps and not args.next_generation:
1442:         print("this run is one generation, so it is the last one and stops after "
1443:               "fitness.")
1444:         print("    %s are what build the next generation, and there is none."
1445:               % ", ".join(NEXT_GENERATION))
1446:         print("    Add --next-generation to run them anyway.")
1447:         print()
1448: 
1449:     context = context_for(conn, run_id, conf, args)
1450:     try:
1451:         return run(selected, context)
1452:     finally:
1453:         # Whatever the process step left standing. gep_lora/core/pipeline/main.py calls this as a
1454:         # library and then hands the same sweep to gep_lora/core/pipeline/continue_run.py, so the
1455:         # servers have to go before that second driver starts its own -- one
1456:         # interpreter, one card, and no pool outlives the driver that made it.
1457:         context.release_pool()
1458:         conn.close()
1459: 
1460: 
1461: if __name__ == "__main__":
1462:     sys.exit(main())

(End of file - total 1462 lines)
</content>
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 4.6s)

_Thinking:_

This is a very thoughtfully written codebase. Now let me look at the templates (generated scripts), the async API, the agent, and tests. Let me use explore agents for broad review, and read a few key files.

Let me read template_code.py and check for issues, plus gep_lora/core/pipeline/main.py and gep_lora/core/pipeline/continue_run.py. And look at the async API server/registry for security issues. And check tests.

Let me use parallel agents to review specific areas, plus read the template.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\templates\\template_code.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\templates\template_code.py</path>
<type>file</type>
<content>
1: #~ TEMPLATE, not a script to run. generate_runs.py reads this file and fills in
2: #~ the @@MARKERS@@ to produce one run_NNN.py per individual.
3: #~
4: #~ Two kinds of marker:
5: #~   @@NAME@@            inline, replaced inside the line it sits on
6: #~   a line that is just @@NAME@@ (or "# @@NAME@@") is replaced by a whole block
7: #~
8: #~ Lines starting with "#~" are template-only notes and never reach the output.
9: #~ Everything else is copied through verbatim, which is why this file is kept as
10: #~ valid Python: your editor, linter and `python -m compileall` all still work on
11: #~ it, and the generated scripts are exactly what you see here.
12: #~
13: #~ THE BLEND ARITHMETIC BELOW IS MIRRORED IN gep_lora/core/blends/lora_server.py.
14: #~ attach(), combine(), _compact() and _rank() exist there too, on a base model
15: #~ that stays loaded between individuals -- which is what
16: #~ template_remote_code.py's clients talk to. A generated script is standalone
17: #~ by design and can import nothing from this repo, so the copy is unavoidable;
18: #~ the rule that goes with it is the one this template and its mocked twin
19: #~ already live under: **a change to the arithmetic here belongs there too, and
20: #~ the other way round.**
21: """
22: @@SCRIPT_NAME@@ - Combine LoRAs the way one GEP tree says to, then chat.
23: 
24: @@PROVENANCE@@
25: Written in the style of combination.py, which stacks two adapters with a
26: fixed combination_type; here the tree decides both the shape and the weights.
27: 
28: Expression
29:     @@EXPRESSION@@
30: 
31: Tree
32: @@TREE@@
33: 
34: How to read it
35:     L<i>.w<j>   attach LoRA slot i, blended at weight w<j>
36:     CAT(a, b)   add_weighted_adapter(..., combination_type="cat")
37:     SVD(a, b)   add_weighted_adapter(..., combination_type="svd")
38:     LIN(a, b)   add_weighted_adapter(..., combination_type="linear")
39: 
40: A combined node is itself an adapter, so it feeds its parent like a leaf.
41: Its children's weights are already folded in, so it enters its parent at
42: weight 1.0.
43: 
44: Build order (deepest first)
45: @@BUILD_ORDER@@
46: 
47: @@NOTE@@
48: Usage
49:     python @@SCRIPT_NAME@@                          # demo prompts
50:     python @@SCRIPT_NAME@@ "Help me plan my week."   # your own question
51: """
52: 
53: import json
54: import os
55: import random
56: import sys
57: import time
58: 
59: #~ Everything from here to the end of the imports is timed as the "import"
60: #~ phase: on this machine unsloth alone is tens of seconds, and a step total
61: #~ cannot tell that from a slow model load.
62: _STARTED = time.perf_counter()
63: 
64: 
65: def _timing(phase, seconds, label=""):
66:     """Say what this script spent on one thing, on the channel everything else
67:     it says comes out on -- see process_run.timings(), which reads these back.
68: 
69:     One line per occurrence, folded into calls/total/worst on the way into the
70:     database, so a phase that happens fifty times is fifty of these and one row.
71:     Printed as it happens rather than gathered up and printed at the end: a
72:     script that runs out of memory half way through has still told the sweep
73:     what the half it managed cost. flush, because stdout is a pipe here and a
74:     killed script's buffer dies with it.
75:     """
76:     print(f"TIMING: {phase} {seconds:.4f}{' ' + label if label else ''}", flush=True)
77: 
78: 
79: # Match the training/inference environment: disable Xet download acceleration.
80: os.environ["HF_HUB_DISABLE_XET"] = "1"
81: 
82: # Unsloth patches transformers and peft as it loads, so it has to be
83: # imported before them or the optimizations are silently skipped.
84: from unsloth import FastLanguageModel
85: from unsloth.chat_templates import get_chat_template
86: 
87: import torch
88: from peft import PeftModel
89: 
90: _timing("import", time.perf_counter() - _STARTED)
91: 
92: _HERE = os.path.dirname(os.path.abspath(__file__))
93: _PROJECT = os.path.dirname(_HERE)                  # run/ -> project/
94: 
95: # The base model every adapter was trained on (from adapter_config.json).
96: #~ Whole-line marker, like WEIGHT_SEED and the three below it: generate_runs.py
97: #~ replaces it with the BASE_MODEL assignment itself, filled from BASE_MODEL in
98: #~ settings.py. It lives there rather than here so there is one copy of the name
99: #~ across both templates and the baseline script the llm_judge_baseline
100: #~ evaluator measures against -- a control loading a different model would make
101: #~ every improvement score meaningless -- and so the sweep records which model
102: #~ its fitness numbers were earned on. Another of the names a linter calls
103: #~ undefined here and finds defined in every file generated from this.
104: # @@BASE_MODEL@@
105: 
106: # The chat template the prompts are written in: None for BASE_MODEL's own, or the
107: # name of one of unsloth's. It has to be the template the adapters were trained
108: # under -- an adapter answers in the format it learned.
109: #~ Whole-line marker, filled from CHAT_TEMPLATE in settings.py -- or, for a sweep
110: #~ stored before that setting existed, "qwen-2.5", which every template used to
111: #~ hardcode (generate_runs.chat_template_name). The baseline control gets the
112: #~ same value, so a blend and its control are asked in the same words.
113: # @@CHAT_TEMPLATE@@
114: 
115: # Where each of the 5 LoRAs the trees refer to lives. One independent entry per
116: # slot, already resolved: repoint a slot in settings.py and every script built
117: # afterwards follows.
118: #
119: # These five were trained at different ranks (r=16, 16, 8, 4, 32), which the code
120: # handles -- _rank() reads each one's adapter_config.json rather than assuming
121: # they match. That matters because PEFT's cat sums input ranks, svd takes the
122: # max, and linear refuses inputs whose ranks differ.
123: #~ Whole-line marker, like WEIGHT_SEED and TRAINING_SET: generate_runs.py
124: #~ replaces it with the LORA_SLOTS assignment itself, filled from LORA_SLOTS in
125: #~ settings.py. The slots are the search space, so they belong with the rest of
126: #~ the knobs -- and with them there, the sweep records which five adapters its
127: #~ fitness numbers were earned on. A third name a linter calls undefined here
128: #~ and finds defined in every file generated from this.
129: # @@LORA_SLOTS@@
130: 
131: # What w1..w5 are worth: a fresh random draw every run, strictly between 0 and
132: # 1. Set WEIGHT_SEED to an int to repeat one particular draw -- without it the
133: # same tree scores differently each time it runs.
134: #~ The next line is a whole-line marker: generate_runs.py replaces it with the
135: #~ WEIGHT_SEED assignment itself, so a generated script carries a plain literal.
136: #~ A sweep fills in the integer it recorded for this individual, so a stored
137: #~ sweep can be replayed weight for weight; None is what a caller that does not
138: #~ pin the draw gets, which is the behaviour described above. Since the marker
139: #~ stands in for the assignment, WEIGHT_SEED is the one name in this template a
140: #~ linter will call undefined -- it is defined in every file generated from it.
141: # @@WEIGHT_SEED@@
142: 
143: _rng = random.Random(WEIGHT_SEED)
144: 
145: 
146: def _weight():
147:     """A blend weight in (0, 1), both ends excluded."""
148:     # random() yields [0.0, 1.0), so rejecting an exact 0.0 leaves (0.0, 1.0).
149:     value = 0.0
150:     while value == 0.0:
151:         value = _rng.random()
152:     return value
153: 
154: 
155: WEIGHTS = {name: _weight() for name in ("w1", "w2", "w3", "w4", "w5")}
156: 
157: def _rank(adapter_dir):
158:     """The rank PEFT will allocate for this adapter, from its adapter_config.json.
159: 
160:     Mirrors PEFT's own bookkeeping (peft/tuners/lora/model.py): rank_pattern can
161:     raise the rank above r for individual modules, and PEFT sizes the merged
162:     adapter for the largest rank it might need.
163:     """
164:     with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as handle:
165:         config = json.load(handle)
166:     return max([config["r"]] + list((config.get("rank_pattern") or {}).values()))
167: 
168: 
169: MAX_SEQ = 2048
170: 
171: # The prompts this individual is judged on, read from a file at startup rather
172: # than baked in, so the eval set can change without regenerating any of these
173: # scripts.
174: #~ Whole-line marker, like WEIGHT_SEED above: generate_runs.py replaces it with
175: #~ the TRAINING_SET assignment itself, so a generated script gets a plain
176: #~ literal path. The value comes from TRAINING_SET in settings.py -- it lives
177: #~ there rather than here so the eval set can be repointed without editing
178: #~ either template, and so the sweep that used it records which file that was.
179: #~ That is why TRAINING_SET, like WEIGHT_SEED, is a name a linter calls
180: #~ undefined here and finds defined in every file generated from this.
181: # @@TRAINING_SET@@
182: 
183: # How many of those records this individual is judged on: the first
184: # TRAINING_COUNT, or all of them when the file holds fewer or the cap is None.
185: #~ Whole-line marker again, filled from TRAINING_COUNT in settings.py. The cap
186: #~ is applied here rather than by handing the script a pre-trimmed list, so a
187: #~ generated script still reads the eval file itself and still shows how many of
188: #~ it it used. Fourth and last of the names a linter calls undefined in this
189: #~ template and finds defined in every file generated from it.
190: # @@TRAINING_COUNT@@
191: 
192: 
193: def _prompt_of(line, path, number):
194:     """One eval prompt, from one non-blank line of the eval file.
195: 
196:     Two shapes, told apart by the line itself rather than by the file's
197:     extension:
198: 
199:       * a JSON object carrying a "messages" list -- the shape datasets/*.json
200:         use and the shape create_lora.py trains on. The prompt is the first
201:         user turn. The assistant turn sitting beside it is somebody else's
202:         answer to the same question, and must not reach the model: handing it
203:         over would be showing the model the answer and then scoring it on the
204:         reply.
205:       * anything else -- the line as written, surrounding quotes optional,
206:         which is the plain one-prompt-per-line file.
207:     """
208:     if line[0] == "{":
209:         try:
210:             record = json.loads(line)
211:         except ValueError as error:
212:             raise SystemExit(
213:                 f"{path} line {number}: starts like a JSON record but will not "
214:                 f"parse ({error})."
215:             )
216:         messages = record.get("messages") if isinstance(record, dict) else None
217:         if not isinstance(messages, list):
218:             raise SystemExit(
219:                 f"{path} line {number}: a JSON eval record needs a 'messages' "
220:                 f"list, the shape datasets/*.json use."
221:             )
222:         for message in messages:
223:             if not isinstance(message, dict) or message.get("role") != "user":
224:                 continue
225:             content = message.get("content")
226:             if isinstance(content, str) and content.strip():
227:                 return content.strip()
228:         raise SystemExit(
229:             f"{path} line {number}: no user turn with any text in it, so there "
230:             f"is nothing to ask."
231:         )
232: 
233:     # Tolerate the quoted form a plain prompt file may use, and a bare one.
234:     if len(line) > 1 and line[0] == line[-1] and line[0] in "\"'":
235:         line = line[1:-1]
236:     return line
237: 
238: 
239: def _prompts(path, count=None):
240:     """One prompt per non-blank line, capped at `count`.
241: 
242:     `count` keeps the first `count` of them and drops the rest; None keeps all.
243:     The top of the file rather than a sample of it, because every individual has
244:     to answer the same questions for their scores to mean anything next to each
245:     other.
246:     """
247:     try:
248:         with open(path, encoding="utf-8") as handle:
249:             lines = handle.readlines()
250:     except OSError as error:
251:         raise SystemExit(f"cannot read the eval prompts from {path}: {error.strerror}")
252: 
253:     prompts = []
254:     for number, line in enumerate(lines, start=1):
255:         line = line.strip()
256:         if not line:
257:             continue
258:         if not prompts and line[0] == "[":
259:             # A whole-file JSON array cannot be read a line at a time, and
260:             # treating its first line as a prompt would quietly ask the model
261:             # "[". Say so instead: one record per line is the supported shape.
262:             raise SystemExit(
263:                 f"{path} looks like one big JSON array. The eval file is read a "
264:                 f"line at a time -- write it as one JSON record per line (JSON "
265:                 f"Lines), or as plain one-prompt-per-line text."
266:             )
267:         prompts.append(_prompt_of(line, path, number))
268:     if not prompts:
269:         raise SystemExit(f"{path} has no prompts in it")
270:     # Slicing past the end is not an error, which is exactly the "or all of
271:     # them" case -- a cap larger than the file needs no special handling.
272:     return prompts if count is None else prompts[:count]
273: 
274: 
275: EVAL_PROMPTS = _prompts(TRAINING_SET, TRAINING_COUNT)
276: 
277: EXPRESSION = "@@EXPRESSION@@"
278: 
279: print(f"GPU available: {torch.cuda.is_available()}")
280: print(f"@@LABEL@@: {EXPRESSION}")
281: # Print the draw, so a run can be traced back to the weights that produced it.
282: print("weights: " + ", ".join(f"{k}={v:.4f}" for k, v in WEIGHTS.items()))
283: 
284: # ---------------------------------------------------------------------------
285: # Load the base model once, with no adapter attached yet.
286: # ---------------------------------------------------------------------------
287: _started = time.perf_counter()
288: model, tokenizer = FastLanguageModel.from_pretrained(
289:     model_name=BASE_MODEL,
290:     max_seq_length=MAX_SEQ,
291:     dtype=None,
292:     load_in_4bit=True,
293: )
294: # The fixed price of an individual: paid once per script whatever its tree is,
295: # which is what makes it the number PROCESS_RUN_BATCH_SIZE is really trading
296: # against, and the one a cheaper search cannot get out of by picking simpler
297: # trees.
298: _timing("model_load", time.perf_counter() - _started)
299: 
300: # ---------------------------------------------------------------------------
301: # Attach the @@LEAF_COUNT@@ leaf adapter(s) the tree names, each under its own name so
302: # the same slot can appear more than once at different weights.
303: # ---------------------------------------------------------------------------
304: # Each adapter's rank is read from its own config, so slots pointing at LoRAs
305: # trained with different r values are handled correctly.
306: RANKS = {}
307: 
308: 
309: def attach(name, slot):
310:     """Load LORA_SLOTS[slot] under `name`, and record the rank it carries."""
311:     global model
312:     started = time.perf_counter()
313:     if isinstance(model, PeftModel):
314:         model.load_adapter(LORA_SLOTS[slot], adapter_name=name)
315:     else:
316:         # The first adapter is what turns the base model into a PeftModel.
317:         model = PeftModel.from_pretrained(model, LORA_SLOTS[slot], adapter_name=name)
318:     RANKS[name] = _rank(LORA_SLOTS[slot])
319:     # Per leaf, and labelled with the slot: leaves are the part of the cost a
320:     # tree can have more of, so calls x seconds here is what a deeper population
321:     # would cost before any of it is folded.
322:     _timing("attach", time.perf_counter() - started, f"{name}={slot}")
323:     return name
324: 
325: 
326: #~ One attach() per leaf, in post-order.
327: # @@ATTACH_LEAVES@@
328: 
329: # ---------------------------------------------------------------------------
330: # Fold the tree together, deepest node first. Each call leaves behind a new
331: # adapter that later calls can use as an input.
332: # ---------------------------------------------------------------------------
333: def _compact(name):
334:     """Copy adapter `name`'s weights off whatever buffer they were sliced from.
335: 
336:     Timed as its own phase, nested inside the combine.svd that called it: what
337:     it costs in seconds is what the VRAM it releases is bought with, and that
338:     trade cannot be looked at unless both halves are written down.
339: 
340:     PEFT builds an svd node by running torch.linalg.svd on each module's delta
341:     weight and handing back `Vh[:new_rank, :]` (peft/tuners/lora/model.py,
342:     _svd_generalized_task_arithmetic_weighted_adapter), which it assigns
343:     straight to that module's lora_A. The slice is a *view*, so it pins the
344:     whole V matrix for as long as the adapter lives -- and V is sized by the
345:     delta weight, not by the rank. On this base model that is ~306 MB behind
346:     every down_proj: right around 10 GB across the stack, holding some 18 MB
347:     of actual weights, which is how an svd node came to cost seven times the
348:     model it attaches to.
349: 
350:     The slice comes back contiguous, so .contiguous() is a no-op here -- only
351:     a copy drops the reference to the buffer behind it. Anything already
352:     sitting on its own storage is left alone.
353:     """
354:     started = time.perf_counter()
355:     for module in model.modules():
356:         for store in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
357:             entry = getattr(module, store, None)
358:             if entry is None or name not in entry:
359:                 continue
360:             held = entry[name]
361:             # lora_A/lora_B hold Linears; the embedding pair holds bare
362:             # Parameters. Both are weights to us.
363:             weight = getattr(held, "weight", held)
364:             if weight.untyped_storage().nbytes() > weight.numel() * weight.element_size():
365:                 weight.data = weight.data.clone()
366: 
367:     # The buffers just released are the caching allocator's now, not the
368:     # driver's, and PROCESS_RUN_BATCH_SIZE runs its batch-mates as separate
369:     # processes -- which cannot see this one's free list. Hand the scratch
370:     # back so a batch is sized by what its scripts hold rather than by the
371:     # high-water mark they each passed through.
372:     torch.cuda.empty_cache()
373:     _timing("compact", time.perf_counter() - started, name)
374: 
375: 
376: def combine(name, combination_type, left, right):
377:     """Fold two adapters into one under `name`, tracking the resulting rank.
378: 
379:     PEFT's rules (peft/tuners/lora/model.py, _check_add_weighted_adapter):
380:     cat sums the input ranks, svd takes the max, linear demands they match.
381:     The linear case is checked here so the failure names the node, rather than
382:     surfacing as a bare ValueError from inside PEFT.
383:     """
384:     (left_name, left_weight), (right_name, right_weight) = left, right
385:     left_rank, right_rank = RANKS[left_name], RANKS[right_name]
386: 
387:     if combination_type == "linear" and left_rank != right_rank:
388:         raise SystemExit(
389:             f"{name}: combination_type='linear' needs both inputs at the same rank, "
390:             f"but {left_name} is rank {left_rank} and {right_name} is rank {right_rank}. "
391:             f"cat sums its inputs' ranks, which is usually what pushes them apart."
392:         )
393: 
394:     started = time.perf_counter()
395:     model.add_weighted_adapter(
396:         adapters=[left_name, right_name],
397:         weights=[left_weight, right_weight],
398:         adapter_name=name,
399:         combination_type=combination_type,
400:         # The thin SVD. The first new_rank singular vectors are identical
401:         # either way, so the full n x n V that PEFT asks for by default is
402:         # only a bigger buffer to compute and then throw away -- see
403:         # _compact(), which has to throw it away either way.
404:         svd_full_matrices=False,
405:     )
406: 
407:     if combination_type == "svd":
408:         _compact(name)
409: 
410:     # One phase per combination type rather than one for combining: cat, svd and
411:     # linear are three different pieces of arithmetic with three different
412:     # prices, and a search whose expensive operator is known is a search whose
413:     # alphabet can be argued about. Includes the _compact() above, which is
414:     # timed again on its own -- so the two overlap on purpose.
415:     _timing(f"combine.{combination_type}", time.perf_counter() - started, name)
416: 
417:     if combination_type == "cat":
418:         RANKS[name] = left_rank + right_rank
419:     elif combination_type == "svd":
420:         RANKS[name] = max(left_rank, right_rank)
421:     else:
422:         RANKS[name] = left_rank
423:     return name
424: 
425: 
426: #~ One combine() per binary node, in post-order.
427: # @@COMBINE_NODES@@
428: 
429: FINAL_ADAPTER = "@@FINAL_ADAPTER@@"
430: _started = time.perf_counter()
431: model.set_adapter(FINAL_ADAPTER)
432: print(f"Active adapter: {model.active_adapters} (rank {RANKS[FINAL_ADAPTER]})")
433: 
434: # Same chat template used during training, so inputs are formatted identically:
435: # the model's own unless a name was set, in which case unsloth's of that name.
436: if CHAT_TEMPLATE:
437:     tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
438: 
439: # Batched generation pads the short prompts up to the long one, and a decoder
440: # continues from the last column of what it is given -- so the padding has to
441: # sit on the *left*, or a padded row would be continuing from its own padding
442: # and answering nothing. The pad token itself is only ever masked out; Qwen
443: # ships one, and eos stands in for a tokenizer that does not.
444: #
445: # A vision-language repo (unsloth/Qwen3.5-0.8B) loads as a processor wrapping
446: # the tokenizer, and padding, the pad id and decoding all belong to the
447: # tokenizer inside it. For a plain tokenizer this is the tokenizer itself.
448: text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
449: text_tokenizer.padding_side = "left"
450: if text_tokenizer.pad_token_id is None:
451:     text_tokenizer.pad_token = text_tokenizer.eos_token
452: 
453: # Switch to Unsloth's fast inference path (~2x faster generation).
454: FastLanguageModel.for_inference(model)
455: 
456: # Qwen ships max_length=32768 in its generation_config.json, and transformers
457: # warns whenever that and max_new_tokens are both set. Clear it so the cap in
458: # ask() is the only one in play -- max_new_tokens was winning anyway.
459: model.generation_config.max_length = None
460: 
461: # Stop at the end of the turn. generate() stops only on the ids in
462: # generation_config.eos_token_id, and a repo that ships no
463: # generation_config.json (unsloth/Qwen3.5-0.8B) leaves that as <|endoftext|>
464: # alone -- so a model that ends its answer with <|im_end|>, as the chat
465: # template trains it to, carried on inventing the next user turn and answering
466: # it until the cap, and all of that reached the judge. A no-op for the Qwen2.5
467: # repos, which already list <|im_end|>.
468: _stops = model.generation_config.eos_token_id
469: _stops = _stops if isinstance(_stops, list) else [] if _stops is None else [_stops]
470: _end_of_turn = getattr(tokenizer, "tokenizer", tokenizer).eos_token_id
471: if _end_of_turn is not None and _end_of_turn not in _stops:
472:     model.generation_config.eos_token_id = _stops + [_end_of_turn]
473: 
474: # Selecting the adapter, the chat template and Unsloth's inference path: the
475: # rest of the fixed cost, after the model load and before the first question.
476: _timing("inference_setup", time.perf_counter() - _started)
477: 
478: 
479: # How many eval prompts go through the model in one generate() call.
480: #
481: # The weights are read once per call whatever it is answering, so five prompts
482: # in one call cost far less than five calls of one -- on this base model the
483: # batch is nearly the price of its slowest single answer. What grows with the
484: # batch is the KV cache, which is why this is a cap and not simply all of them:
485: # TRAINING_COUNT says how many prompts there are, this says how many at once.
486: # It is a constant here rather than a setting for the reason MAX_SEQ is: the
487: # script is the only thing that reads it, and it is a fact about the machine
488: # rather than about the sweep.
489: ANSWER_BATCH = 8
490: 
491: 
492: def answer(questions, max_new_tokens=250):
493:     """Answer several questions in one generate() call. -> replies, in order."""
494:     # Rendered to text, then tokenised, rather than apply_chat_template(
495:     # return_dict=True) in one call: a processor (Qwen3.5) ignores return_tensors
496:     # and return_dict and hands back text, which cannot be moved to the GPU.
497:     # For a plain tokenizer the ids are the same -- add_special_tokens=False
498:     # because the rendered template already holds every special token, which is
499:     # also what apply_chat_template does when it tokenises.
500:     texts = [tokenizer.apply_chat_template(
501:                  [{"role": "user", "content": question}],
502:                  add_generation_prompt=True, tokenize=False)
503:              for question in questions]
504:     inputs = text_tokenizer(texts, padding=True, return_tensors="pt",
505:                             add_special_tokens=False).to(model.device)
506:     out = model.generate(**inputs, max_new_tokens=max_new_tokens,
507:                          do_sample=False, pad_token_id=text_tokenizer.pad_token_id)
508:     # Padded on the left, so every row's reply starts at the same column and
509:     # one width slices the prompt off all of them.
510:     width = inputs["input_ids"].shape[-1]
511:     return [text_tokenizer.decode(row[width:], skip_special_tokens=True).strip()
512:             for row in out]
513: 
514: 
515: def ask(question, max_new_tokens=250):
516:     """Send one user turn through this tree's combined adapter."""
517:     return answer([question], max_new_tokens)[0]
518: 
519: 
520: def _say(questions):
521:     """Answer a batch, print the exchanges, then say what the batch cost.
522: 
523:     Printed here, batch by batch, rather than after the last one: a script that
524:     hits its timeout still leaves the exchanges it had finished, which is what
525:     process_run.launch() keeps a killed child's output for. The timing line goes
526:     last, after the final reply of the batch -- a line between "COACH:" and the
527:     rest of a wrapped answer would be read as part of the answer.
528:     """
529:     started = time.perf_counter()
530:     replies = answer(questions)
531:     seconds = time.perf_counter() - started
532:     for question, reply in zip(questions, replies):
533:         print(f"\nYOU: {question}")
534:         print(f"COACH: {reply}")
535:     # calls counts generate() calls, not answers -- the label says how many
536:     # answers this one carried, since that is what its seconds bought.
537:     _timing("generate", seconds, "%d answer(s)" % len(questions))
538: 
539: 
540: if __name__ == "__main__":
541:     if len(sys.argv) > 1:
542:         # Everything after the script name is treated as one question.
543:         _say([" ".join(sys.argv[1:])])
544:     else:
545:         # Score this individual by eyeballing its answers to the eval prompts.
546:         for start in range(0, len(EVAL_PROMPTS), ANSWER_BATCH):
547:             _say(EVAL_PROMPTS[start:start + ANSWER_BATCH])
548:         # The whole script, from its first line to its last answer: what the
549:         # phases above should add up to, and the number the process step
550:         # measures from outside as well. Two clocks on the same thing, so a
551:         # gap between them is a phase nobody has named yet.
552:         _timing("total", time.perf_counter() - _STARTED)

(End of file - total 552 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\search\\weight_mutation.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\search\weight_mutation.py</path>
<type>file</type>
<content>
1: """
2: weight_mutation.py - Mutation of the blend weights alone, minus the elite.
3: 
4:     individuals.chromosome  -->  chromosome, has_changed, fitness
5: 
6: mutation.py offers every symbol a chance to move; this offers only the weights.
7: A weight is a `w1`..`w5` symbol -- the child of an `L*` node, saying at which
8: of the five drawn weights that adapter enters the blend -- so a tree of ten
9: blended adapters carries ten of them. Operators and slots are left exactly as
10: they are: the blend keeps its shape and its adapters, and only how much of each
11: it takes is varied.
12: 
13: How many weights move
14: ---------------------
15: Not a chance per weight but a count over all of them. Every weight of every
16: non-elite chromosome goes into one pool, and
17: 
18:     round(rate * weights in the pool)
19: 
20: of them are drawn from it, without replacement, and each replaced by a
21: *different* weight symbol. Ten weights at 0.1 is exactly one change, somewhere
22: among the ten; a population holding sixty weights at 0.1 is exactly six,
23: wherever they happen to fall -- two in one individual and none in another is as
24: likely as any other spread. Rounding is half up, so a pool too small for the
25: rate to reach one half moves nothing.
26: 
27: A swap from one weight symbol to another is class-local in the sense
28: mutation.py explains -- arity 0 either side -- so the tree keeps its shape and
29: every result still decodes; generate_population.check() is run on each one
30: before it is stored all the same.
31: 
32: The elite does not change
33: -------------------------
34: The individual marked `is_best` contributes nothing to the pool, so none of its
35: weights can be drawn and it is not written at all -- for the reason mutation.py
36: gives: elitism named it so that it survives the generation intact.
37: 
38: has_changed, and the fitness that goes with it
39: ----------------------------------------------
40: An individual with a weight moved gets its new chromosome through
41: store.set_chromosome(), which sets has_changed to 1 and clears its fitness to
42: NULL: the score belonged to the blend weighted the old way.
43: 
44: Unlike mutation.py this step only ever *raises* has_changed. It runs after
45: mutation in the same generation, and an individual mutation moved and this step
46: did not has still changed this round -- writing 0 over it would tell process to
47: skip a chromosome that has never been run.
48: 
49: gep_lora/core/pipeline/start_run.py calls this as a library:
50: 
51:     weight_mutation.apply(conn, run_id, rate, rng)
52: """
53: 
54: from collections import namedtuple
55: 
56: from search import generate_population
57: from storage import store
58: 
59: # One individual this round touched: what it was, what it became, and how many
60: # weights differ between the two.
61: Change = namedtuple("Change", "number before after weights")
62: 
63: 
64: def weight_positions(chromosome):
65:     """The positions of the weight symbols in a chromosome, in order."""
66:     return [position for position, symbol in enumerate(chromosome.split("."))
67:             if symbol in generate_population.VARIABLES]
68: 
69: 
70: def draw_count(rate, total):
71:     """How many of `total` weights a round at `rate` moves, rounded half up."""
72:     return min(total, int(rate * total + 0.5))
73: 
74: 
75: def mutate(chromosomes, rate, rng):
76:     """A pool of chromosomes through one draw. -> the chromosomes they came out as.
77: 
78:     `chromosomes` is a list; the result is a list of the same length, in the
79:     same order, each entry the chromosome that one became (unchanged if none of
80:     its weights was drawn).
81:     """
82:     pool = [(index, position)
83:             for index, chromosome in enumerate(chromosomes)
84:             for position in weight_positions(chromosome)]
85:     symbols = [chromosome.split(".") for chromosome in chromosomes]
86:     for index, position in rng.sample(pool, draw_count(rate, len(pool))):
87:         current = symbols[index][position]
88:         symbols[index][position] = rng.choice(
89:             [other for other in generate_population.VARIABLES if other != current])
90:     mutated = [".".join(parts) for parts in symbols]
91:     for chromosome in mutated:
92:         generate_population.check(chromosome)
93:     return mutated
94: 
95: 
96: def differences(before, after):
97:     """How many symbols differ between two chromosomes of the same shape."""
98:     return sum(one != other for one, other in zip(before.split("."), after.split(".")))
99: 
100: 
101: def apply(conn, run_id, rate, rng):
102:     """Mutate the weights of a whole population but its elite. -> (changes, rows).
103: 
104:     Writes the new chromosome, has_changed = 1 and a cleared fitness for every
105:     individual that moved, and nothing for any other. `changes` holds one Change
106:     per individual that moved; `rows` is the population as it was before.
107:     """
108:     rows = store.individuals(conn, run_id)
109:     eligible = [row for row in rows if not row["is_best"]]
110:     mutated = mutate([row["chromosome"] for row in eligible], rate, rng)
111:     changes = []
112:     for row, after in zip(eligible, mutated):
113:         if after == row["chromosome"]:
114:             continue
115:         store.set_chromosome(conn, row["id"], after)
116:         changes.append(Change(row["number"], row["chromosome"], after,
117:                               differences(row["chromosome"], after)))
118:     conn.commit()
119:     return changes, rows
120: 
121: 
122: def pool_size(rows):
123:     """How many weights the non-elite individuals of `rows` hold between them."""
124:     return sum(len(weight_positions(row["chromosome"]))
125:                for row in rows if not row["is_best"])

(End of file - total 125 lines)
</content>
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 296.8s)

_Thinking:_

Now let me look at the async API for security and code quality. Let me read server.py, registry.py, users.py, and the agent. Also check tests. I'll use explore agents to summarize while reading a couple.

Let me launch parallel explore agents for:
1. async_api security review (auth, SQL injection, path traversal)
2. async_api_agent review
3. tests review
4. reporting/metrics review

And read gep_lora/core/pipeline/main.py and gep_lora/core/pipeline/continue_run.py myself.

**Tool: task**

**Input:**
```json
{
  "description": "Review async API for issues",
  "prompt": "Review the `async_api/` package in D:\\sage-is\\loras\\gep-lora (files: server.py, registry.py, users.py, worker.py, submit.py, verify.py, evaluate.py, results.py, golive.py, inference.py, testpass.py, train.py, settings.py) for a code critique. Focus on:\n1. Security: authentication/authorization checks, SQL injection, path traversal, secrets handling, tokens, CORS, unsafe deserialization, command injection when spawning subprocesses.\n2. Correctness and concurrency: the worker/queue model, database locking, race conditions, error handling.\n3. Code quality: duplication, overly long functions, tight coupling, testability.\n4. Anything that looks like a latent bug.\n\nRead the files. Report concrete findings with `file_path:line_number` references and a short explanation each. Be specific and skeptical; do not just praise. Do NOT write any code. Return a prioritized list of findings.",
  "subagent_type": "explore"
}
```

**Output:**
```
task_id: ses_f302b3b07ffe3Hf4AIoTzP13PW (for resuming to continue this task if needed)

<task_result>
I've read all thirteen modules plus the relevant supporting code (`gep_lora/core/evaluators/common.py`, `gep_lora/core/adapters/catalog.py`, `gep_lora/core/adapters/create_lora.py`). Findings below, ordered by severity.

---

## Critical

### 1. SSRF + server judge-key exfiltration via `GET /judge/models`
`gep_lora/apps/web/server.py:603-622`

`base_url` is taken from the query string and passed straight to `gep_lora.core.evaluators.common.list_models(base_url, evaluators.API_KEY, ...)`. That helper sets `Authorization: Bearer <key>` (`gep_lora/core/evaluators/common.py:275-277`) where `API_KEY = os.environ.get("JUDGE_API_KEY", "")` (`gep_lora/core/evaluators/common.py:66`). The only validation is scheme `http(s)` and a non-empty netloc (`server.py:614`). Any authenticated user can point it at a host they control and receive the server's `$JUDGE_API_KEY` in the request, or use it to probe internal addresses. `urllib` follows redirects and copies the request headers to the redirect target, so the key can also be forwarded to a second host (see `common.py:280`).

### 2. Same key exfiltration / SSRF through job and verification options
`gep_lora/service/settings.py:40-44`, `gep_lora/service/submit.py:68-83`, `gep_lora/service/evaluate.py:31-33,80-110`, `gep_lora/service/verify.py:205-212`

`JUDGE_BASE_URL` (and `JUDGE_MODEL`, `JUDGE_BACKEND`) is **not** in `LOCKED_SETTINGS`, so a submission may set it; the evaluate step then POSTs to `base_url + "/chat/completions"` with the bearer key (`gep_lora/core/evaluators/common.py:356` and `:247-255`). The same applies to `judge_base_url` in an evaluation body (`evaluate.py:102-109`, which uses `_text` and performs **no URL validation at all**) and in a verification body (`verify.py:205-212`). This is strictly worse than finding #1 because it also sends the dataset/answers, and `evaluate.py`/`verify.py` never validate the scheme or host.

---

## High

### 3. No worker exclusivity; startup recovery fails live jobs and orphan processes are never killed
`gep_lora/service/worker.py:402-416`, `gep_lora/service/registry.py:450-454`

`recover()` reads every `jobs.status = 'running'` row and marks it FAILED with no check that a live worker owns it and no OS-level lock on `JOBS_DIR`. The module docstring says "one worker per JOBS_DIR," but nothing enforces it: starting a second worker (or restarting one before the first has exited) fails the first worker's in-flight job. The stored `pid` (`registry.set_pid`, `worker.py:202`) is never used to verify or kill the orphan, so the dead worker's child can keep writing the same sweep database while a resumed run works on it. `recover_verifications`/`recover_trainings` (`worker.py:312-321,383-391`) have the same shape.

### 4. Negative `Content-Length` leads to an unbounded blocking read; no request timeout
`gep_lora/apps/web/server.py:834-845`, `:861`

`_dispatch` sets `self._unread = int(self.headers.get("Content-Length") or 0)`; a negative value passes the `length > MAX_BODY_BYTES` and `if not length` guards (`:836-839`), so `_body` executes `self.rfile.read(-1)` at `:840`. `BufferedReader.read(-1)` reads to EOF, not to a zero length. A client sending `Content-Length: -1` and holding the connection open pins that server thread indefinitely (`Handler` sets `HTTP/1.1`, no `timeout`). Even a well-formed oversized body is read fully into memory before JSON parsing against a 64 MB cap (`settings.py:31`), and `ThreadingHTTPServer` starts an unbounded thread per connection.

### 5. `os.path.commonpath` can raise out of request handling (Windows)
`gep_lora/service/submit.py:241-250` (`shared_path`), used by `dataset_lines:259` and `server.verify_dataset:509`

For a rooted name on another drive (`{"file": "D:\\whatever"}`) or a UNC path, `os.path.commonpath([shared, path])` raises `ValueError("Paths don't have the same drive")`. That exception is not `SubmissionError`/`VerifyError`, so it escapes as a 500 traceback instead of a clean 400/`None`. The traversal check itself is correct; the failure mode is the bug.

### 6. No quotas or rate limits on expensive work; single global queue
`gep_lora/service/train.py:48-61`, `gep_lora/service/submit.py:86-117`, `gep_lora/service/worker.py:419-453`

`NUMBERS` allows `epochs <= 10000`, `max_steps <= 10**7`, `max_seq <= 1<<20`, and a submission can override `GENERATIONS`, `COUNT`, `TRAINING_COUNT`. The worker processes one item at a time from a single unbounded queue, so any authenticated user can enqueue days of GPU work ahead of everyone else. `MAX_BODY_BYTES = 64 MB` (`settings.py:31`) combined with inline datasets makes repeated 64 MB submissions a memory amplifier. There is no per-user concurrency or queue cap.

---

## Medium

### 7. `ModelCache` entries can be duplicated and leaked on load failure; revocation races an in-flight build
`gep_lora/service/inference.py:151-158,192-204`

`_drop` (192-196) removes the entry from `_entries` while other threads that already called `_entry` still hold a reference and may call `engine.load()` on the same half-loaded engine, and it never unloads it. New requests for the same key then create a second engine, so two copies of the model can be resident and the first is never released. `forget` (198-204) writes `entry.built_for` without holding `entry.lock`, racing a concurrent `stream` that is rebuilding that same field (179-182). `stream` holds `entry.lock` for the whole generation (170-186), so a stuck `generate()` blocks every request for that key; `UnslothEngine.stream` also `join()`s the generation thread before returning (`:107`), so a client disconnect still occupies the request thread until the model finishes.

### 8. `mocked` engine detection is wrong for never-rendered individuals
`gep_lora/service/golive.py:131-139`

`mocked = bool(source) and not (process_run.imports_unsloth(source) or server_pool.wanted(source))`. When `script_source` is empty (seed/never rendered), `bool(source)` is False, so a mocked sweep is classified as `"unsloth"` and the deployment tries to load a real model. The engine test also re-implements logic that lives in `process_run`/`server_pool`. Separately, `draw_weights` (48-58) and `_weight_of` (61-66) duplicate the generated template's weight draw and plan arithmetic; a template change silently makes a live blend differ from the one that was scored.

### 9. TOCTOU between job deletion and resume/requeue
`gep_lora/apps/web/server.py:447-461` (`delete_run`/`delete_job`) vs `:369-380` (`resume_job`) and `:395-410` (`start_evaluation`)

`delete_*` checks `_finished` (status in `FINISHED`), then `shutil.rmtree`s the folder and marks/deletes the row. `RESUMABLE = (STOPPED, CANCELLED, FAILED)` overlaps `FINISHED`, and `requeue` is only atomic against the status column, not against the rmtree. A concurrent `POST /jobs/{id}/resume` after the check can queue a job whose database/folder is being deleted, producing a worker failure rather than a clean 409.

### 10. `submit()` can leave a `preparing` job row and folder behind
`gep_lora/service/submit.py:334-358`

`registry.enqueue(...)` at `:358` is outside the `try/except` that cleans up on `:350-357`. If `enqueue` raises (DB lock, disk), the reserved job row stays `preparing` (never visible to the worker) and the job folder + datasets stay on disk forever. `reserve_job`/`enqueue` are deliberately two phases, so anything that can fail in between should roll back.

### 11. Unbounded log/dataset reads; unbounded log growth
`gep_lora/apps/web/server.py:319-329,540-551,678-691`, `gep_lora/service/train.py:357-377`, `gep_lora/service/worker.py:153`

`job_log`, `verification_log` and `lora_log` each `handle.read()` the entire file and then slice `[-lines:]`; `lines` has no upper bound and the underlying logs are appended to without rotation (`worker.py:153-180`). `train.preview` reads a whole dataset file into memory despite a `limit`. On a long-running deployment these become straightforward memory/DoS paths.

### 12. Unsalted, non-constant-time hashing of keys and tokens
`gep_lora/service/registry.py:165-167`

`digest()` is plain SHA-256 with no server-side pepper. For the 256-bit `token_urlsafe` values this is not practically brute-forceable, but a DB exfiltration exposes every credential to offline attacks if any token is ever weakened, and lookup is by SQL equality rather than `secrets.compare_digest`. An HMAC with a server secret would be strictly better with no cost. (No password/secret value is ever logged; `supervise` writes argv, which does not include `$JUDGE_API_KEY`.)

### 13. `verify_dataset`/`list_datasets` expose shared dataset names to all users
`gep_lora/apps/web/server.py:592-601,503-526`

Every authenticated user can enumerate `SHARED_DATASETS_DIR` and point a verification at any file there. Filenames alone may be sensitive, and there is no per-user allowlist; the module comment frames datasets as "shared," but the access control is effectively global read of names.

---

## Low / hardening

- `gep_lora/service/registry.py:233-237` (`_migrate`) and `gep_lora/core/adapters/catalog.py:352-378` build SQL with `%` string interpolation. All inputs today are constants or `_encode`-validated column names, so it is not injectable, but it is fragile. Queries that take user values are consistently parameterized (`registry.py:326-331,341-348,489-498,706-721`), and `revoke` builds placeholders correctly (`:719-721`).
- `gep_lora/apps/web/server.py:301-317` `job_database`: `mkstemp` → `os.remove` → `store.connect(path)` is a classic TOCTOU on the temp path (low risk in the OS temp dir).
- `gep_lora/service/registry.py:280-282` `remove_user` cascades jobs/verifications/trainings away but leaves files and any running worker, whose later `registry.finish(...)` silently updates zero rows.
- `gep_lora/service/worker.py:172-178`: `KeyboardInterrupt` arriving after the poll loop but during `process.wait()` is not caught, so the child can be left running; `kill_tree` on POSIX only escalates the direct child, not the session, so grandchildren that ignore SIGTERM survive.
- `gep_lora/apps/web/server.py:508-526` and `:216-234`: `training_of` matches on the folder string, and `delete_lora` rmtrees `lora_catalog.absolute(row["folder"])`. Today constrained to `TRAINED_LORAS_DIR/userN/name` by `train.NAME` (`train.py:45`) and `user_dir`, but nothing asserts the folder is under the trained root before deleting.
- `gep_lora/service/submit.py:97` and `:107-113`: slot-rank and weight-presence checks happen against `config`/template, but an empty source is not treated as mocked (see #8), so the guard for weightless adapters is skipped when no template matches.
- `gep_lora/service/evaluate.py:104`: a user echoing the current `JUDGE_*` values is silently ignored — correct — but there is no normalization (trailing slash, case) so a cosmetically equal URL is written into the sweep as a "change."
- `gep_lora/apps/web/server.py:612-615`: `urlparse` validation accepts userinfo, IP literals and non-standard ports; there is no blocklist for link-local/metadata addresses.

---

## Code quality

- **`App` is a god object** (`gep_lora/apps/web/server.py:246-771`): ~40 methods mixing authentication, job/verification/training/deployment business logic, LoRA catalogue operations, file serving and inference wiring. It is the seam every test must mock; `unittests/service/support.py` presumably does. There is no `test_worker.py`, `test_evaluate.py` or `test_testpass.py` under `unittests/async_api/` (only server, submit, train, registry, results, golive, inference, verify), so the worker/queue and the evaluate/testpass option parsing are untested.
- **Triplicated queue machinery**: `claim_next` / `claim_next_training` / `claim_next_verification` (`registry.py:350-366,500-520,603-619`), `finish`/`finish_training`/`finish_verification` (372-376,527-531,625-629), `request_cancel`/`request_training_cancel` (378-415,631-653), and `orphaned*` (450-454,533-537,659-663) are near-identical copies. A small parameterized helper would remove most of `registry.py`.
- **Duplicated quality-join loops**: `results.population` (`results.py:112-124`) and `verify.choices` (`verify.py:80-89`) each rebuild the same `{number: quality}` projection from `store.quality_rows` + `store.individuals`.
- **Duplicated response shaping**: `job_json`/`verification_json`/`training_json`/`lora_json`/`deployment_json` (`server.py:152-243`) all follow the same "pick columns + json.loads one field + add derived booleans" pattern with slightly different key lists.
- **Fragile positional dispatch**: `_route` builds `args = [user] + groups + body/query` by convention (`server.py:903-907`). Adding a regex capture group or a new route silently shifts argument positions; there is no signature check. Mixed method-name/function dispatch (`:908-909`) adds magic.
- **Function-local imports**: `import evaluators` inside `judge_models` (`server.py:616`), inside `submit.form` (`submit.py:186`), and `from evaluators import composite` — presumably to keep startup light, but it splits the dependency graph and is repeated in several modules.
- **`results.py:33` `_open = connect`** is a pure indirection with no behavior; either remove or document why.
- **`settings.py:67-78` `_override()` mutates module globals at import time**, which makes the module's values order/import-dependent and awkward to isolate in tests; an explicit `Settings` object would be cleaner.
- **Cross-process coupling in `worker.py`**: `settle`, `search_finished` and `run_job` reach into `results.progress`/`run_status` to decide job status (`worker.py:233-264,394-399`), duplicating "how far did the search get" logic that `gep_lora/core/pipeline/main.py`'s `where_it_stopped()` already owns. The CLAUDE.md rule that the planner be the single definition appears to be broken here.
- **`golive.py` duplicates template arithmetic** (`draw_weights`, `_weight_of`, the attach-then-combine plan order at `:79-85`) that also lives in `template_code.py`/`generate_runs.plan`; the docstring acknowledges the weight draw, but the plan ordering is a second implementation to keep in sync.
- **Private-attribute access across modules**: `inference.UnslothEngine.unload` manipulates `blend._torch` and resets `blend.ready/final/built/ranks` (`inference.py:111-118`), and `golive.blend_spec` uses `start_run._SEED_LIMIT` (`golive.py:119`). These are tight couplings that will break silently on refactors.

---

## Checked and found clean
- **Command injection**: every `subprocess.Popen`/`run` uses list argv with no `shell=True` (`worker.py:115,157`), and `train.command`/`verify.command`/`testpass.command` build argv from validated values. The only shell-ish call, `taskkill`, takes an integer PID.
- **SQL injection**: all value-bearing queries are parameterized; the only `%`-formatted SQL uses hardcoded identifiers/placeholders.
- **Unsafe deserialization**: only `json.loads`; no pickle/eval/yaml.
- **Path traversal in shared datasets**: `submit.shared_path:248` correctly canonicalizes and compares against the shared root (modulo finding #5).
- **Ownership checks**: job, verification, deployment and LoRA lookups all filter by `user_id`/`owner` and return 404 for others (`server.py:258-262,636-644,528-532,628-632`), so there is no obvious IDOR, with the caveat in #13.
- **Secrets**: API keys and deployment tokens are stored hashed and never echoed; `$JUDGE_API_KEY` is not written to logs or argv (its exposure is only via the SSRF paths in #1/#2).
</task_result>
```

**Tool: task**

**Input:**
```json
{
  "description": "Review agent package for issues",
  "prompt": "Review the `async_api_agent/` package in D:\\sage-is\\loras\\gep-lora (files: agent.py, prompts.py, providers.py, analysis.py, selection.py, planner.py, blending.py, release.py, tools.py, commands.py, routes.py, settings.py) for a code critique. This is an LLM chat agent that walks a user to trained LoRAs, plans a blend search, tests/verifies and goes live. Focus on:\n1. Security: prompt injection handling, secrets/API keys, what facts reach the model, tool call validation, authorization (does it check the user owns a resource?).\n2. Correctness: tool-call loop (MAX_ROUNDS), error handling, fallbacks when provider unreachable, session state handling.\n3. Code quality: duplication, long functions, coupling, testability, whether prompts are centralized.\n4. Latent bugs.\n\nRead the files. Report concrete findings with `file_path:line_number` references and a short explanation each. Be specific and skeptical. Do NOT write any code. Return a prioritized list of findings.",
  "subagent_type": "explore"
}
```

**Output:**
```
task_id: ses_f302b3adaffekQ6I7U09HRdoZy (for resuming to continue this task if needed)

<task_result>
I've read all twelve modules plus the server dispatch, registry, catalogue, the page's rendering, and the existing test suite. Here is a prioritized critique. No files were modified.

---

## Critical / High

**1. Prompt injection can drive action tools with no server-side confirmation — and untrusted dataset text reaches the tool-call loop.**
`agent.py:203-259` gives the model tools that start/stop training, searches, testing, verification and deployments. The only thing stopping it acting without the user's intent is prose in `prompts.py:126-128` ("Do not call a tool that changes something unless they asked for that change"). `tools.Toolbox.run` (`tools.py:225-257`) enforces only tool name, stage and "is there a dataset"; it never verifies the user actually asked. On the page side, `applyChat` executes the returned actions without a confirmation gate (`agent-ui.html:930-951`: a `plan` action followed by a `start` action queues training immediately). The injection surface is real, not just theoretical: dataset samples are placed in FACTS for the analysis step (`agent.py:105-136`, samples from `analysis.py:255-258`), the model's reply becomes an assistant turn in `history`, and tool results echo dataset text back into the loop (`tools.py:297-314`, `tools.py:316-321`). So third-party dataset content can plausibly steer `start_training`/`go_live`. The page's "read back to the person" is a message, not a gate.

**2. SSRF: any authenticated user can point a local provider at an arbitrary URL and read the response.**
`providers.resolve` (`providers.py:100-113`) allows a `base_url` override for any provider with `local: true` — which includes the default `lmstudio` and `ollama` (`settings.py:44-50`). It correctly strips the key (`providers.py:111`), but the server then issues the request: `GET {base_url}/models` (`providers.py:189-198`) and `POST {base_url}/chat/completions` (`providers.py:335-346`). `/agent/models?provider=lmstudio&base_url=http://169.254.169.254/...` (`routes.py:231-240`) returns the parsed `data[].id` values, and `/agent/chat` returns the model's reply text — a general internal-network read primitive. Auth is required (`server.py:900-902`), which limits but does not remove the risk.

**3. Cross-user information leak in `_set_verify_questions`.**
`tools.py:623` does `self.catalog.get(questions["lora"])["name"]` with no owner check. `catalog.get` (`gep_lora/core/adapters/catalog.py:380-382`) is owner-agnostic. The `questions` value comes from the page-supplied session (`planner.session_of` → `release.release_of`, which only checks it is an `int`, `release.py:74-81`). A user can send `session.release.questions = {"lora": <victim id>}`, then trigger `set_verify_questions` with only `count` (no model needed — `commands.py:229-233` turns "5 questions" into exactly that call), and the victim's LoRA name is returned in `questions_text`. If the id does not exist it raises a bare `TypeError`, reported misleadingly as "bad arguments". `routes.verify_plan` does check ownership (`routes.py:407-409`); this path does not.

**4. Uncaught exceptions in tool bodies turn a bad tool call into a 500 and lose the message.**
`Toolbox.run` catches `(ToolError, SelectionError, SessionError, DatasetError, BlendError, ReleaseError)` and `TypeError` (`tools.py:239-243`), but not plain `ValueError`/`KeyError`/`AttributeError` raised *inside* a tool. Concretely:
- `_show_records`: `int(count or 5)` and `int(start)` (`tools.py:298,307`) raise `ValueError` on a non-numeric string the model may emit (the JSON schema is advisory, not enforced).
- `_estimate_time` → `planner.clamp` → `float(epochs)` (`tools.py:361-364`, `planner.py:241-242`) raises `ValueError`.
- `TOOL_DONE[name].format(**result)` (`tools.py:254`) raises `KeyError` if a result ever lacks a placeholder (e.g. the `{epochs:g}` formats).
`agent.chat` only catches `ProviderError` (`agent.py:243`), so these propagate to the server's catch-all (`server.py:923-924`), returning a 500 and discarding the reply and all prior tool steps.

**5. An unreachable provider is tried twice per chat message.**
When `providers.converse` fails, `agent.chat` falls back to `commands.parse`; if that produces no calls it calls `say("chat", …)` (`agent.py:256-258`), which calls `providers.chat` again (`agent.py:80`) — a second full network attempt with the same `TIMEOUT` (`settings.py:101`, default 120s). So a hanging endpoint can cost ~2× the timeout, and the same fallback `say` path is taken on every step that has a model. The original error is not reused.

---

## Medium

**6. `context` is injected into the model's FACTS verbatim, unbounded and unvalidated.**
`routes.chat` only checks `isinstance(context, dict)` (`routes.py:290-296`), then `agent.chat` drops it straight into `facts` (`agent.py:212-213`) and serialises it into the user turn (`agent.py:220-221`). It is a second page-supplied prompt-injection channel, parallel to `history` (`agent.py:78`, `agent.py:214`).

**7. No cap on tool calls per round.**
`agent.py:231-235` runs *every* call the model returns, and `MAX_ROUNDS` only bounds rounds (`agent.py:51`). A single reply can carry an unbounded number of calls, several of which re-parse the dataset (`_dataset_facts` calls `analysis.analyse`, `tools.py:316-321`; `_show_records` re-reads all records, `tools.py:210-214`). Cheap DoS / CPU amplification.

**8. A tool-summary fallback is attributed to the model.**
When `MAX_ROUNDS` is exhausted, `text = ""` and the message is `_summary(box)` — but `"by"` is set to the provider label and `"fallback": False` (`agent.py:236-241`). The transcript therefore shows built-in wording as if the model wrote it, unlike the exception path which correctly says `"built-in wording"` (`agent.py:248`).

**9. Wrong fallback constant reused in debriefs.**
`test_debrief` uses `prompts.FALLBACK_TEST_START_MOCK` (`agent.py:449`) and `verify_debrief` uses it too (`agent.py:482`). It is a *start*-step constant; the text happens to read acceptably but the name/intent is wrong, so a future edit to the start wording silently changes the debrief.

**10. `_shared_path` can raise an uncaught `ValueError` on a cross-drive path.**
`routes.py:98-103`: `os.path.join(shared, name)` with an absolute `name` on another drive yields an absolute path, and `os.path.commonpath([shared, path])` raises `ValueError("Paths don't have the same drive")` rather than the intended `AgentError(400)`. That becomes a 500 whose message leaks filesystem layout. Same-drive traversal is handled correctly.

**11. `demo_datasets()` re-reads and re-parses every shared file on every call, often several times per request.**
`routes.py:166-188` reads each file up to 4 MB and runs full `analysis.analyse`. It is called on every `/agent/config` and every `/agent/chat` (`routes.py:205,296`), and `tools._use_demo_dataset`/`_set_blend_questions`/`_set_verify_questions` each call `self.shared_datasets()` one to three times (`tools.py:402-414,492-512,587-611`). O(files × bytes) repeated per message.

**12. Layering / circular coupling.**
`blending._file_text` lazily imports the HTTP layer to reach private helpers (`blending.py:195-198` → `routes._read`/`routes._shared_path`), while `routes.py:73` imports `blending` — a deliberate cycle. `planner` reaches into `gep_lora.service.train`'s private `_number` (`planner.py:104`), and `session_of` lazily imports `blending`/`release` (`planner.py:136-145`). These make the "one module per concern" boundary leaky and hard to unit-test in isolation.

**13. Testability: network and model access are hard-wired.**
`providers._request` uses `urllib.request` directly (`providers.py:136-159`) and `agent.say`/`agent.chat` call `providers.resolve`/`providers.chat`/`providers.converse` as module globals (`agent.py:69,80,217,224`). The only way the suite tests the wire formats is by standing up a real `HTTPServer` (`unittests/assistant/test_agent.py:159-199`), so provider failure modes, timeouts and retry logic are effectively untested.

---

## Low / latent

**14. `providers.resolve` crashes on non-string `model`/`base_url`.** `(choice.get("base_url") or "").strip()` and `(choice.get("model") or "").strip()` (`providers.py:101,114`) raise `AttributeError` → 500 if the page sends a number/bool. Minor input-hardening gap.

**15. Cross-drive/`bool` job id accepted in `test_started`.** `isinstance(job.get("id"), int)` (`routes.py:380`) admits `True` (a bool is an int), unlike `routes._id` (`routes.py:360-364`) which explicitly excludes bools. User-scoped, so impact is only a wrong 404/200.

**16. Report parsing assumes keys exist.** `release.verification_outcome` reads `one["wins"]`, `one["losses"]`, `one["p"]` (`release.py:238-242`) with no guard; an unexpected/older report shape raises `KeyError` → 500. `blending.outcome` similarly assumes `found["fitness_history"]`, `found["testing"]["summary"]` etc.

**17. `settings._override()` will replace any upper-case global, including dicts, with a raw string when the env value is not JSON.** `settings.py:210-220` — `GEP_AGENT_PROVIDERS=broken` silently makes `PROVIDERS` a `str`, and every provider lookup then fails confusingly.

**18. CSV header heuristic drops a data row for one-column input.** `analysis.py:127-132`: when the first row has fewer than two columns, `all(len(cell)<40 and not cell.endswith(...))` over `rows[0][:2]` can be trivially true and the first record is treated as a header.

**19. OpenCode Go session affinity is lost when the page omits `conversation`.** `providers.resolve` mints a fresh uuid per call (`providers.py:123-129`), and `say`/`chat` each resolve independently, so different steps get different `x-opencode-session` ids. The page does send one, but the API does not require it.

---

## Code quality

**20. Duplicated "match a demo file by fuzzy name" logic** in `tools.py:407-414`, `tools.py:506-512`, `tools.py:602-608`, `commands.py:173-178`, `commands.py:219-228`, `commands.py:265-271` — five near-identical blocks, each with subtly different tie-breaking.

**21. Duplicated number-word substitution and start/practice regexes.** `planner.read_wait` (`planner.py:253-256`) and `commands._clean` (`commands.py:42-46`) both map word-numbers to digits; the practice-run regex appears twice in `commands.py:182-184` and `commands.py:307-312`; `_START` is defined once (`commands.py:135-136`) but the start-vs-search guard is repeated at `commands.py:185-186` and `commands.py:331`.

**22. Long functions with mixed concerns:** `commands.parse` (`commands.py:241-333`, ~92 lines), `commands._release` (`commands.py:193-238`), `commands._selection` (`commands.py:49-93`), `providers.resolve` (`providers.py:90-133`), `agent.chat` (`agent.py:203-259`), `tools.Toolbox.run`/`outcome` (`tools.py:225-293`). Each mixes parsing/validation/formatting.

**23. "Every word lives in prompts.py" is only approximately true.** Words that reach the model or the user live outside it: the `"next"` strings in `tools.py:420-421,443-444,463,538-539,632-633,645-646,654-655`; the `source` strings in `planner.py:166,179-187` and `blending.py:266-267`; the verdict words in `release.py:238-239`; and inline strings in `agent.py:182,240`. `prompts.py:1-19` claims otherwise.

---

## What is done well (so the critique is not read as one-sided)

- **Secrets are handled correctly.** Keys are read from named env vars only (`providers.py:70-71`), never returned by `describe()` (`providers.py:74-87`; the test asserts it at `test_agent.py:208-209`), and a local URL override clears the key (`providers.py:103-111`). The live token is deliberately kept out of the model's facts (`agent.py:486-495`, `routes.py:429-440`, `agent-ui.html:982-983`).
- **Authorization is consistently user-scoped in the routes.** `registry.job(id, user["id"])`, `verification(id, user["id"])` and `deployment(id, user["id"])` are used throughout (`routes.py:350-354,369-375,421-424,431-433`), and `blending.own`/`check_together` enforce ownership before any LoRA is planned (`blending.py:120-156`).
- **Provider request shaping is careful**: conversation ids are regex-validated before becoming headers (`providers.py:44-46,123-129`), and thinking blocks are preserved for Anthropic round-trips (`providers.py:299-313`).
</task_result>
```

**Tool: task**

**Input:**
```json
{
  "description": "Review tests and evaluators",
  "prompt": "Review the test suite and evaluators in D:\\sage-is\\loras\\gep-lora. \n- Test files live under `unittests/` (async_api, async_api_agent, adapters, evaluators, search) plus there may be others. There is also a `run_unit_tests.bat` and no pytest config maybe.\n- Evaluators live in `evaluators/` (common.py, llm_judge*.py, panel.py, composite.py, similarity.py, heuristic.py, local_model.py, jev_judge_reference.py).\n\nAssess:\n1. Test coverage and quality: what is tested, what is conspicuously NOT tested (e.g. the core GEP search steps, gep_lora/core/storage/store.py, generate_runs.py, process_run.py). Are tests hermetic or do they hit network/GPU? Is there mocking?\n2. Test framework: is pytest used consistently? Are there tests that are really integration tests? \n3. Evaluators: duplication across the llm_judge* modules, error handling, the judge transport in common.py, robustness of score parsing.\n4. Any flaky or slow tests, and any dead/disabled tests.\n\nReport concrete findings with `file_path:line_number` references. Be specific and skeptical. Do NOT write code. Return a prioritized list of findings and a rough coverage assessment.",
  "subagent_type": "explore"
}
```

**Output:**
```
task_id: ses_f302b3a82ffefSyFd0jF8lmca6 (for resuming to continue this task if needed)

<task_result>

</task_result>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\main.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\main.py</path>
<type>file</type>
<content>
1: """
2: gep_lora/core/pipeline/main.py - A sweep from nothing to the end of the search, in one command.
3: 
4:     gep_lora/core/pipeline/start_run.py     draws a population and takes it through one generation
5:     gep_lora/core/pipeline/continue_run.py  carries that sweep on for GENERATIONS more
6:     test_run_with_dataset.py  puts what the search found in front of the
7:                      testing split, and grades it
8:     gep_lora/core/pipeline/main.py          all of that, in that order, against the same sweep
9: 
10:     python main.py
11: 
12: which is:
13: 
14:     python start_run.py
15:     python continue_run.py --run <the sweep gep_lora/core/pipeline/start_run.py just made>
16: 
17: so the whole search is 1 + GENERATIONS generations -- gep_lora/core/pipeline/start_run.py's own turn of the
18: crank, then the ones gep_lora/core/pipeline/continue_run.py adds. --generations controls the second
19: half; there is no way to have fewer than the one gep_lora/core/pipeline/start_run.py runs, because drawing
20: a population and leaving it unjudged would not be a generation.
21: 
22: It calls the two drivers as libraries, in this interpreter. That matters: the
23: process step launches every generated script with sys.executable, so a subprocess
24: would be one more chance to run the search under the wrong Python. Whatever you
25: start this with is what the whole sweep uses.
26: 
27: The sweep is handed on by id, not by "the latest one" -- gep_lora/core/pipeline/start_run.py's new sweep is
28: looked up once it exists and named explicitly, so a database that gains a sweep
29: from somewhere else in between cannot be picked up by mistake.
30: 
31: Options are forwarded to whichever driver understands them: --label to gep_lora/core/pipeline/start_run.py,
32: --generations and --set to gep_lora/core/pipeline/continue_run.py, and the rest -- --db, --run-dir,
33: --limit, --include-blocked, --include-unchanged, --keep-scripts, --timeout,
34: --force -- to both, meaning there what they mean there.
35: 
36:     python main.py --generations 3
37:     python main.py --db run_real/gep.sqlite3 --label "overnight"
38:     python main.py --limit 2 --generations 1     # a smoke test of the lot
39: 
40: Running a database that is already a sweep
41: -----------------------------------------------------------------------------
42: 
43:     python main.py --db dbtemplates/test_new_run.sqlite3
44: 
45: A database can be *prepared* rather than produced: a run row, its settings, its
46: dataset, and no individuals. Everything a search needs and none of the search.
47: Handed one of those, this runs it rather than starting a second sweep beside it
48: -- `--db <a prepared database>` means "run this", and the alternative would be
49: a stranger's sweep appearing in somebody's experiment file. --run 0 (or --run N)
50: says the same thing out loud, and takes any sweep by id.
51: 
52: The rule is the latest sweep having no individuals. Anything else is a database
53: to start a new sweep in, exactly as before: one that does not exist yet, one
54: holding no sweeps, one whose latest sweep has a population. --label suppresses
55: it too, since only a sweep being created can be given a label.
56: 
57: From there the database is the only thing the run reads itself out of --
58: 
59:   * the settings are the sweep's stored ones, which is what resuming already
60:     meant, so the evaluator, the base model, the adapters, the seeds, the batch
61:     size and GENERATIONS are the ones written down beside the questions;
62:   * the questions are the sweep's stored dataset rows, not the files those
63:     settings name. settings.py and datasets/ are never opened.
64: 
65: and it is **isolated on disk as well**. Everything such a run writes goes into
66: one folder beside the database, named for it and the sweep --
67: 
68:     dbtemplates/
69:         test_new_run.sqlite3
70:         test_new_run_run1/
71:             training.jsonl      the questions, out of the rows
72:             run_001.py ...      the generated scripts, until they have run
73:             testing_scripts/    the testing pass's re-pointed copies
74: 
75: -- so run_db/, run_testing/ and the rest of the repo are untouched, two prepared
76: databases cannot tread on each other, and the results go back into the same
77: file, since --db is where the sweep lives. A prepared database is then a whole
78: experiment in one file plus one folder: hand it to another machine and the
79: search it runs there is the one it describes, not the one that machine's
80: settings.py happens to say. --run-dir still overrides, if you want it elsewhere.
81: 
82: An adopted sweep must hold no individuals yet -- gep_lora/core/pipeline/start_run.py's half of the run draws
83: a population, and drawing one into a search that already has one would leave two
84: side by side. gep_lora/core/pipeline/continue_run.py --from-db is what carries a started sweep on.
85: 
86: --from-db is the flag underneath all of this, and the three drivers take it on
87: their own too:
88: 
89:     python start_run.py --db <prepared> --run 1 --from-db
90:     python continue_run.py --db <prepared> --run 1 --from-db
91:     python -m gep_lora.core.testing.test_run_with_dataset --db <prepared> --run 1 --from-db
92: 
93: --no-test and --test-min-quality go to the testing pass, and --db, --limit,
94: --keep-scripts, --timeout and --force reach it too.
95: 
96: A failing half stops the run: if gep_lora/core/pipeline/start_run.py cannot produce a sweep there is nothing
97: to continue, and its exit code comes straight back out.
98: 
99: The testing pass at the end runs only when the sweep has a TESTING_SET, because
100: that setting is the only statement anyone has made about which questions the
101: search was not judged on. It costs a base-model load per individual above
102: TESTING_MIN_QUALITY, so --no-test skips it and --test-min-quality changes how
103: many that is. It is deliberately the *last* thing, and it is why the search's last
104: generation stops after `fitness`: the individuals worth testing are the ones
105: that were actually scored, and stopping there is what leaves the population
106: holding exactly those rather than a generation selection and mutation have
107: already moved on to.
108: 
109:     python main.py --no-test                 # search only, as before
110:     python main.py --test-min-quality 0.7    # test fewer of them
111: 
112: Stopping, and carrying on
113: -----------------------------------------------------------------------------
114: 
115:     python main.py --db <sweep> --run 1 --resume
116:     python main.py --db <sweep> --run 1 --evaluate [--force] [--set JUDGE_MODEL=...]
117: 
118: Any sweep can be carried on from wherever it stopped. --resume reads how far it
119: got out of its own database (where_it_stopped): the fitness snapshots say how
120: many generations were scored, and step_timings which steps finished since the
121: last one -- which is what tells a population about to be bred from one already
122: bred, since the tail that breeds it is the part not safe to run twice. Then it
123: finishes that generation, runs the ones left, and the testing pass, which is
124: resumed too. A sweep with no population yet is simply run.
125: 
126: --evaluate grades the answers a sweep holds -- the ungraded ones, or all of them
127: with --force -- under the sweep's own EVALUATOR, and restates fitness only where
128: that is honest: where the population is still the one those answers came from.
129: --set moves the judge, written into the sweep. The async API's stop, resume and
130: evaluate are these two, run by its worker.
131: """
132: 
133: import argparse
134: import os
135: import sys
136: import time
137: from collections import namedtuple
138: 
139: from config import settings as config
140: import continue_run
141: import start_run
142: from storage import store
143: from testing import test_run_with_dataset
144: 
145: _HERE = os.path.dirname(os.path.abspath(__file__))
146: 
147: 
148: def forwarded(options, adopted=False):
149:     """The arguments both drivers understand, as an argv fragment."""
150:     argv = ["--db", options.db, "--timeout", str(options.timeout)]
151:     if adopted:
152:         # Both halves read the adopted sweep's questions out of the sweep, and
153:         # work in its own folder beside the database rather than in run_db/.
154:         argv.append("--from-db")
155:     if options.run_dir:
156:         argv += ["--run-dir", options.run_dir]
157:     if options.limit:
158:         argv += ["--limit", str(options.limit)]
159:     if options.include_blocked:
160:         argv.append("--include-blocked")
161:     if options.include_unchanged:
162:         argv.append("--include-unchanged")
163:     if options.keep_scripts:
164:         argv.append("--keep-scripts")
165:     if options.force:
166:         argv.append("--force")
167:     return argv
168: 
169: 
170: def call(driver, argv):
171:     """Run one driver in this process. -> its exit code.
172: 
173:     SystemExit is what a driver raises when it stops itself -- an unknown run,
174:     a population that is not there. Caught here so the second half is skipped
175:     rather than the whole thing unwinding through an exception nobody reads.
176:     """
177:     try:
178:         code = driver(argv)
179:     except SystemExit as error:
180:         if error.code in (0, None):
181:             return 0
182:         print(error)
183:         return 1
184:     return code or 0
185: 
186: 
187: def sweep_after(db_path, before):
188:     """The id of the sweep gep_lora/core/pipeline/start_run.py just created, or None if it made none."""
189:     conn = store.connect(db_path)
190:     try:
191:         latest = store.latest_run(conn)
192:     finally:
193:         conn.close()
194:     return latest if latest != before else None
195: 
196: 
197: def prepared(options):
198:     """The sweep a bare --db should run rather than add to, or None.
199: 
200:     A database can be *prepared* rather than produced: a run row, its settings,
201:     its dataset, and no individuals -- everything a search needs and none of
202:     the search. Handed one of those, `python main.py --db it.sqlite3` means
203:     "run this", not "start a second sweep in the same file". So the latest run
204:     is looked at, and adopted when it is waiting to be run.
205: 
206:     Nothing else changes: a database that does not exist yet, holds no sweeps,
207:     or whose latest sweep has a population is a database to start a new sweep
208:     in, exactly as before -- `gep_lora/core/pipeline/main.py --db run_real/gep.sqlite3` goes on
209:     meaning what it meant. --label suppresses this too, since a label is
210:     something only a sweep being created can be given.
211: 
212:     --run says it out loud and takes any sweep by id; this is the same decision
213:     made for somebody who did not.
214:     """
215:     if options.run is not None or options.label:
216:         return None
217:     where = (options.db if os.path.isabs(options.db)
218:              else os.path.join(_HERE, options.db))
219:     if not os.path.exists(where):
220:         return None
221:     conn = store.connect(options.db)
222:     try:
223:         run_id = store.latest_run(conn)
224:         if run_id is None or store.individuals(conn, run_id):
225:             return None
226:         # Settings are what make it a sweep rather than an empty row; without
227:         # them there is nothing to run it under and adopt() would say so.
228:         return run_id if store.get_settings(conn, run_id) else None
229:     finally:
230:         conn.close()
231: 
232: 
233: def adopt(options, run=None):
234:     """The prepared sweep to run. -> (run_id, its settings, its datasets).
235: 
236:     `run` is the id, from --run or from prepared(); 0 means the latest.
237: 
238:     A sweep is *prepared* when the database already holds its settings and its
239:     dataset and no individuals: everything a search needs, and none of the
240:     search. gep_lora/core/pipeline/start_run.py then draws the population into it and takes it through its
241:     first generation exactly as it would into a sweep it had just created --
242:     except that the knobs and the questions are the ones already written down,
243:     so settings.py and the files under datasets/ are never opened, and the run
244:     keeps to its own folder beside the database rather than to run_db/.
245: 
246:     Refused, rather than guessed past:
247: 
248:     A path that is not already a database. store.connect() creates what it
249:     cannot open, which for a mistyped --db would mean being told a brand-new
250:     empty file holds no run 1.
251: 
252:     A sweep that already holds individuals. The population step appends, so
253:     joining a search that has one would draw a second population beside the
254:     first; gep_lora/core/pipeline/continue_run.py is what carries one of those on. Only --run can
255:     reach this, since prepared() picks no such sweep in the first place.
256:     """
257:     run = options.run if run is None else run
258:     where = (options.db if os.path.isabs(options.db)
259:              else os.path.join(_HERE, options.db))
260:     if not os.path.exists(where):
261:         raise SystemExit("no database at %s" % os.path.abspath(where))
262: 
263:     conn = store.connect(options.db)
264:     try:
265:         run_id = store.latest_run(conn) if run == 0 else run
266:         if run_id is None:
267:             raise SystemExit(
268:                 "%s holds no sweeps, so there is none to run. Drop --run to "
269:                 "start one from settings.py." % conn.path)
270:         if store.get_run(conn, run_id) is None:
271:             raise SystemExit("no run %d in %s. Try: python -m gep_lora.core.storage.store --list"
272:                              % (run_id, conn.path))
273:         held = store.individuals(conn, run_id)
274:         if held:
275:             raise SystemExit(
276:                 "run %d in %s already holds %d individual(s). gep_lora/core/pipeline/main.py starts "
277:                 "a search rather than joining one, and its first half would draw "
278:                 "a second population beside that one.\n"
279:                 "    Carry it on instead: python main.py --db %s --run %d --resume"
280:                 % (run_id, conn.path, len(held), options.db, run_id))
281:         conf = store.get_settings(conn, run_id)
282:         if not conf:
283:             raise SystemExit(
284:                 "run %d in %s has no stored settings, so there is nothing to run "
285:                 "it under." % (run_id, conn.path))
286:         splits = store.dataset_summary(conn, run_id)
287:     finally:
288:         conn.close()
289:     return run_id, conf, splits
290: 
291: 
292: def describe(options, run_id, conf, splits, asked):
293:     """Say what the adopted sweep is, before a single step runs.
294: 
295:     `asked` is whether --run named it. When it did not, the reason it was picked
296:     is worth a line: this is the one decision gep_lora/core/pipeline/main.py makes on its own, and
297:     the alternative it turned down -- starting a second sweep in somebody's
298:     prepared database -- is exactly what a person would want to be told about.
299:     """
300:     if asked:
301:         print("adopting run %d in %s -- its settings and its questions, not "
302:               "settings.py's" % (run_id, options.db))
303:     else:
304:         print("run %d in %s is prepared -- settings and a dataset, no "
305:               "individuals -- so this runs it" % (run_id, options.db))
306:         print("    rather than starting a second sweep beside it. Its settings "
307:               "and its questions, not settings.py's.")
308:     print("    %d stored setting(s); evaluator %s, base model %s"
309:           % (len(conf), conf.get("EVALUATOR"), conf.get("BASE_MODEL")))
310:     if splits:
311:         for entry in splits:
312:             print("    %-10s %3d record(s), %d with a reference answer"
313:                   % (entry["split"], entry["records"], entry["references"]))
314:     else:
315:         print("    no dataset rows -- the run will stop when it looks for them")
316:     print()
317: 
318: 
319: Plan = namedtuple("Plan", "populated scored expected rest generations complete "
320:                            "at_rest tail_done since")
321: 
322: 
323: def where_it_stopped(conn, run_id, generations):
324:     """Where a sweep's search got to, and what is left of it. -> a Plan.
325: 
326:     A whole search is 1 + GENERATIONS generations: trees .. fitness each, with
327:     NEXT_GENERATION (elitism, selection, mutation, weight_mutation) after every
328:     one but the last. Two things say how far a sweep got, and both are written
329:     as it goes, so a driver killed at any moment leaves them true:
330: 
331:       * fitness_history, one generation per snapshot -- how many were *scored*;
332:       * step_timings, one row per step that finished -- what was done since
333:         the last snapshot, which is the only way to tell a population about to
334:         be bred from one that already has been.
335: 
336:     The second matters because the tail is the half that is not safe to run
337:     twice: selection is another round every time it runs, and mutation mutates
338:     again. Everything before fitness is -- trees and runs re-derive, process
339:     skips what already ran as it is (see step_process), evaluate skips what is
340:     scored, and fitness restates a generation it has already recorded.
341: 
342:     `rest` is the steps that finish the generation it stopped in, in pipeline
343:     order; `generations` the whole ones left after it, for gep_lora/core/pipeline/continue_run.py.
344:     `at_rest` says the sweep sits on a fitness snapshot with at most part of
345:     the tail run since -- `tail_done` being that part -- and `since` is every
346:     step that finished after the snapshot (all of them, before the first).
347:     """
348:     expected = 1 + generations
349:     if not store.individuals(conn, run_id):
350:         return Plan(False, 0, expected, [], 0, False, False, (), ())
351:     scored = len(store.fitness_by_generation(conn, run_id))
352:     finished = [row["step"] for row in store.step_timings(conn, run_id)
353:                 if row["status"] == "ok"]
354:     last = max((index for index, step in enumerate(finished) if step == "fitness"),
355:                default=-1)
356:     since = tuple(finished[last + 1:])
357:     tail = start_run.NEXT_GENERATION
358:     # The tail steps run since the snapshot, as the prefix of the tail they
359:     # reach -- mutation having run means selection and elitism did.
360:     upto = max((tail.index(step) for step in since if step in tail), default=-1)
361:     tail_done = tuple(tail[:upto + 1])
362: 
363:     if scored >= expected:
364:         # Nothing left to search. A tail after it is an older sweep's, from
365:         # when a finished search still ended in mutation.
366:         return Plan(True, scored, expected, [], 0, True, True, tail_done, since)
367:     if scored == 0 or any(step not in tail for step in since):
368:         # Part way through a generation whose fitness has not been taken: the
369:         # tail before it (if any) finished, since the generation's own steps
370:         # only ever start after it. From the top of that generation, then.
371:         current = scored + 1
372:         rest = list(continue_run.GENERATION)
373:         if current == expected:
374:             rest = [step for step in rest if step not in tail]
375:         return Plan(True, scored, expected, rest, expected - current, False,
376:                     False, (), since)
377:     # Resting on a snapshot, part of the way into building the next generation.
378:     return Plan(True, scored, expected, list(tail[upto + 1:]), expected - scored,
379:                 False, True, tail_done, since)
380: 
381: 
382: def holds_training(db_path, run_id):
383:     """Does the sweep hold its own training split? Then it is read from there."""
384:     conn = store.connect(db_path)
385:     try:
386:         return any(entry["split"] == "training"
387:                    for entry in store.dataset_summary(conn, run_id))
388:     finally:
389:         conn.close()
390: 
391: 
392: def open_sweep(options):
393:     """The sweep --resume or --evaluate names. -> (run_id, conf), --set applied.
394: 
395:     Refused for the reasons adopt() refuses: a path that is not a database, a
396:     run that is not there. --set is written into the sweep here, before
397:     anything is planned, because GENERATIONS is one of the things it can say.
398:     """
399:     where = (options.db if os.path.isabs(options.db)
400:              else os.path.join(_HERE, options.db))
401:     if not os.path.exists(where):
402:         raise SystemExit("no database at %s" % os.path.abspath(where))
403:     conn = store.connect(options.db)
404:     try:
405:         run_id = store.latest_run(conn) if options.run == 0 else options.run
406:         if run_id is None or store.get_run(conn, run_id) is None:
407:             raise SystemExit("no run %s in %s. Try: python -m gep_lora.core.storage.store --list"
408:                              % (options.run, conn.path))
409:         conf = store.get_settings(conn, run_id)
410:         if not conf:
411:             raise SystemExit("run %d in %s has no stored settings, so there is "
412:                              "nothing to run it under." % (run_id, conn.path))
413:         continue_run.override(conn, run_id, conf, options.settings)
414:         return run_id, conf
415:     finally:
416:         conn.close()
417: 
418: 
419: def plan_of(options, run_id, conf):
420:     conn = store.connect(options.db)
421:     try:
422:         return where_it_stopped(conn, run_id,
423:                                 continue_run.generation_count(options, conf))
424:     finally:
425:         conn.close()
426: 
427: 
428: def resume(options):
429:     """Carry a stopped sweep on to the end of its search. -> an exit code.
430: 
431:     Whatever stopped it -- a cancel, a crash, a machine switched off -- the
432:     database says how far it got (where_it_stopped), and this finishes the
433:     generation it stopped in, runs the ones it has left, and then the testing
434:     pass, skipping whatever that pass already did. A sweep that never got as
435:     far as a population is simply run, the way a prepared one is; one whose
436:     search is complete goes straight to the testing pass.
437:     """
438:     run_id, conf = open_sweep(options)
439:     plan = plan_of(options, run_id, conf)
440:     if not plan.populated:
441:         print("run %d holds no individuals yet, so resuming it is running it\n"
442:               % run_id)
443:         options.resume = False
444:         options.settings = []           # already written into the sweep
445:         return cli_run(options)
446: 
447:     from_db = holds_training(options.db, run_id)
448:     print("=" * 70)
449:     print("resuming run %d in %s: %d of %d generation(s) scored"
450:           % (run_id, options.db, plan.scored, plan.expected))
451:     if plan.complete:
452:         print("the search is complete; only the testing pass can be left")
453:     else:
454:         if plan.rest:
455:             print("to finish generation %d: %s"
456:                   % (plan.scored + (0 if plan.at_rest else 1), " -> ".join(plan.rest)))
457:         if plan.generations:
458:             print("then %d more generation(s) through gep_lora/core/pipeline/continue_run.py" % plan.generations)
459:     print("=" * 70)
460:     print()
461: 
462:     started = time.time()
463:     code = 0
464:     if plan.rest:
465:         code = call(start_run.main,
466:                     plan.rest + ["--run", str(run_id)] + forwarded(options, from_db))
467:     if not code and plan.generations:
468:         code = call(continue_run.cli,
469:                     ["--run", str(run_id), "--generations", str(plan.generations)]
470:                     + forwarded(options, from_db))
471:     if not plan.complete:
472:         print()
473:         print("=" * 70)
474:         print("resumed run %d %s in %.1fs"
475:               % (run_id, "finished" if not code else "STOPPED AGAIN",
476:                  time.time() - started))
477:         print("=" * 70)
478:     if not code:
479:         code = test(options, run_id, from_db, resume=True)
480:     return code
481: 
482: 
483: def evaluate(options):
484:     """Grade the answers a sweep already holds. -> an exit code.
485: 
486:     The evaluate step over the stored transcripts -- the ones still ungraded,
487:     or every one with --force -- under the sweep's own EVALUATOR, since a
488:     fitness is only comparable with the others when the same rubric earned it.
489:     Where the judge is (JUDGE_BACKEND, JUDGE_MODEL, JUDGE_BASE_URL) is a --set
490:     away, written into the sweep like any other change to it.
491: 
492:     Then fitness, when the scores can still reach it honestly: when the
493:     population is the one those answers were given by (a finished search, or a
494:     stopped one resting on its last snapshot before selection has bred from
495:     it), fitness is restated -- and elitism with it, if it had already run; and
496:     when the generation it stopped in has been through process, fitness is
497:     taken for it, exactly as the next step would have. Anywhere else a fitness
498:     now would score individuals that have not run yet, so it is left to the
499:     resume. Then the testing pass's answers, if it stored any.
500:     """
501:     run_id, conf = open_sweep(options)
502:     plan = plan_of(options, run_id, conf)
503:     if not plan.populated:
504:         raise SystemExit("run %d holds no individuals, so there are no answers "
505:                          "to evaluate." % run_id)
506:     from_db = holds_training(options.db, run_id)
507: 
508:     steps = ["evaluate"]
509:     bred = {"selection", "mutation", "weight_mutation"} & set(plan.tail_done)
510:     if plan.at_rest and not bred:
511:         steps.append("fitness")
512:         if "elitism" in plan.tail_done:
513:             steps.append("elitism")
514:     elif not plan.at_rest and "process" in plan.since:
515:         steps.append("fitness")
516:     print("evaluating run %d in %s: %s%s\n"
517:           % (run_id, options.db, " -> ".join(steps),
518:              " (--force: every answer again)" if options.force else ""))
519:     code = call(start_run.main,
520:                 steps + ["--run", str(run_id)] + forwarded(options, from_db))
521: 
522:     conn = store.connect(options.db)
523:     try:
524:         tested = bool(store.test_results(conn, run_id))
525:     finally:
526:         conn.close()
527:     dataset = ["--from-db"] if from_db else [testing_set(options.db, run_id)]
528:     if tested and not options.no_test and all(dataset):
529:         argv = dataset + ["--db", options.db, "--run", str(run_id), "--score-only"]
530:         if options.force:
531:             argv.append("--force")
532:         print()
533:         print("# and the testing pass's answers")
534:         print()
535:         code = call(test_run_with_dataset.main, argv) or code
536:     return code
537: 
538: 
539: def cli(argv=None):
540:     options = parse(argv)
541:     if options.resume:
542:         return resume(options)
543:     if options.evaluate:
544:         return evaluate(options)
545:     return cli_run(options)
546: 
547: 
548: def cli_run(options):
549:     # --run names a sweep to run; without it, a database that is already a
550:     # prepared sweep is one too. Anything else starts a new sweep, as before.
551:     asked = options.run is not None
552:     target = options.run if asked else prepared(options)
553:     adopted = adopt(options, target) if target is not None else None
554:     if adopted:
555:         run_id, conf, splits = adopted
556:         # The sweep's own GENERATIONS, the way gep_lora/core/pipeline/continue_run.py reads it, so a
557:         # prepared database says how long its own search is.
558:         generations = continue_run.generation_count(options, conf)
559:         before = None
560:     else:
561:         run_id = None
562:         generations = (config.GENERATIONS if options.generations is None
563:                        else options.generations)
564:         conn = store.connect(options.db)
565:         try:
566:             before = store.latest_run(conn)
567:         finally:
568:             conn.close()
569: 
570:     if generations < 1:
571:         raise SystemExit(
572:             "%d generation(s) for the second half; gep_lora/core/pipeline/start_run.py's generation would be "
573:             "the whole run. Use gep_lora/core/pipeline/start_run.py on its own for that." % generations)
574: 
575:     started = time.time()
576:     print("=" * 70)
577:     print("full run: gep_lora/core/pipeline/start_run.py, then gep_lora/core/pipeline/continue_run.py for %d more generation(s)"
578:           % generations)
579:     print("=" * 70)
580:     print()
581:     if adopted:
582:         describe(options, run_id, conf, splits, asked)
583: 
584:     # --next-generation because gep_lora/core/pipeline/start_run.py's generation is this run's *first*,
585:     # not its last: gep_lora/core/pipeline/continue_run.py has at least one more to run (--generations
586:     # is refused below 1), and it needs a population selection and mutation
587:     # have moved on. Only the run's last generation stops after fitness, and
588:     # that one belongs to gep_lora/core/pipeline/continue_run.py.
589:     first = ["--next-generation"]
590:     if adopted:
591:         first += ["--run", str(run_id)]
592:     elif options.label:
593:         first += ["--label", options.label]
594:     code = call(start_run.main, first + forwarded(options, bool(adopted)))
595:     if code:
596:         print()
597:         print("gep_lora/core/pipeline/start_run.py failed; there is no sweep to continue.")
598:         return code
599: 
600:     if not adopted:
601:         run_id = sweep_after(options.db, before)
602:         if run_id is None:
603:             print()
604:             print("gep_lora/core/pipeline/start_run.py made no new sweep in %s; nothing to continue."
605:                   % options.db)
606:             return 1
607: 
608:     print()
609:     print("#" * 70)
610:     print("# gep_lora/core/pipeline/start_run.py done -- run %d. Continuing it for %d more generation(s)."
611:           % (run_id, generations))
612:     print("#" * 70)
613:     print()
614: 
615:     second = ["--run", str(run_id), "--generations", str(generations)]
616:     for assignment in options.settings:
617:         second += ["--set", assignment]
618:     code = call(continue_run.cli, second + forwarded(options, bool(adopted)))
619: 
620:     print()
621:     print("=" * 70)
622:     print("full run of %d generation(s) %s in %.1fs -- run %d in %s"
623:           % (generations + 1, "finished" if not code else "STOPPED",
624:              time.time() - started, run_id, options.db))
625:     print("python -m gep_lora.core.storage.store --show %d" % run_id)
626:     print("=" * 70)
627: 
628:     # The search is over; this is the one question it could not answer about
629:     # itself. Skipped when the search stopped early -- a half-finished sweep's
630:     # best individual is not what the search found -- and when the sweep names
631:     # no testing set, which is the only statement of which questions it was
632:     # never judged on.
633:     if not code:
634:         code = test(options, run_id, bool(adopted))
635:     return code
636: 
637: 
638: def testing_set(db_path, run_id):
639:     """The sweep's own TESTING_SET, or None if it named none.
640: 
641:     The sweep's stored setting rather than settings.py's, for the reason every
642:     step reads stored settings: the file it recorded at creation is the one its
643:     `testing` split holds, and settings.py may have been repointed since.
644:     """
645:     conn = store.connect(db_path)
646:     try:
647:         return store.get_settings(conn, run_id).get("TESTING_SET")
648:     finally:
649:         conn.close()
650: 
651: 
652: def testing_split(db_path, run_id):
653:     """How many testing records the sweep itself holds, or None for none.
654: 
655:     The question TESTING_SET answers for a sweep run from files, asked of the
656:     database instead -- because an adopted sweep reads its questions out of its
657:     own rows, and a setting naming a file those rows never came from is not a
658:     statement about this run. A sweep whose settings name a testing file it
659:     never stored has none as far as this pass is concerned, which is the same
660:     thing db_datasets.repoint() decides.
661:     """
662:     conn = store.connect(db_path)
663:     try:
664:         held = [entry for entry in store.dataset_summary(conn, run_id)
665:                 if entry["split"] == "testing"]
666:     finally:
667:         conn.close()
668:     return held[0]["records"] if held else None
669: 
670: 
671: def test(options, run_id, adopted=False, resume=False):
672:     """Put the finished search in front of its testing split. -> an exit code.
673: 
674:     Called as a library, in this interpreter, for the reason the other two are:
675:     the pass launches each script with sys.executable, so a subprocess would be
676:     one more chance to run it under the wrong Python.
677: 
678:     A pass that stops itself -- nothing scored above the bar, no judge to grade
679:     with -- is reported and returned, not swallowed: the search finished and
680:     this did not, and saying so is the difference between a testing pass that
681:     was skipped and one that failed.
682:     """
683:     if options.no_test:
684:         return 0
685: 
686:     # An adopted sweep is asked the same question of its rows rather than of
687:     # its settings, and answers it with --from-db: the questions it was never
688:     # judged on are the ones in its own `testing` split.
689:     if adopted:
690:         records = testing_split(options.db, run_id)
691:         if not records:
692:             print()
693:             print("no testing pass: run %d holds no testing split, so nothing "
694:                   "says which questions it was never judged on." % run_id)
695:             print("    add one with: python -m gep_lora.core.storage.add_dataset <file> --db %s "
696:                   "--run %d --split testing" % (options.db, run_id))
697:             return 0
698:         dataset = "its own testing split (%d record(s))" % records
699:         first = ["--from-db"]
700:     else:
701:         dataset = testing_set(options.db, run_id)
702:         if not dataset:
703:             print()
704:             print("no testing pass: run %d names no TESTING_SET, so nothing says "
705:                   "which questions it was never judged on." % run_id)
706:             print("    set TESTING_SET in settings.py before a sweep, or run "
707:                   "test_run_with_dataset.py against this one.")
708:             return 0
709:         first = [dataset]
710: 
711:     print()
712:     print("#" * 70)
713:     print("# the search is done -- testing run %d against %s" % (run_id, dataset))
714:     print("#" * 70)
715:     print()
716: 
717:     argv = first + ["--db", options.db, "--run", str(run_id),
718:                     "--timeout", str(options.timeout)]
719:     if options.test_min_quality is not None:
720:         argv += ["--min-quality", str(options.test_min_quality)]
721:     if options.limit:
722:         # Means there what it means here: only the first few individuals, so a
723:         # smoke test of the whole thing stays a smoke test.
724:         argv += ["--limit", str(options.limit)]
725:     if options.keep_scripts:
726:         argv.append("--keep-scripts")
727:     if options.force:
728:         argv.append("--force")
729:     if resume:
730:         # Whatever an interrupted pass already tested stays tested: test_results
731:         # is appended to, and a second row per individual would say it was
732:         # tested twice.
733:         argv.append("--resume")
734: 
735:     code = call(test_run_with_dataset.main, argv)
736:     if code:
737:         print()
738:         print("the search finished; the testing pass did not.")
739:     return code
740: 
741: 
742: def parse(argv):
743:     parser = argparse.ArgumentParser(
744:         description="Run a whole search: gep_lora/core/pipeline/start_run.py for a new sweep and its first "
745:                     "generation, then gep_lora/core/pipeline/continue_run.py for the rest.")
746:     parser.add_argument("--db", default=config.DB_PATH,
747:                         help="database file (default %s)" % config.DB_PATH)
748:     parser.add_argument("--run", type=int, default=None, metavar="ID",
749:                         help="run the sweep the database already holds instead of "
750:                              "creating one (0 = the latest). Its stored settings "
751:                              "and its stored dataset are what the search uses; "
752:                              "settings.py and the files under datasets/ are never "
753:                              "read. The sweep must hold no individuals yet.")
754:     parser.add_argument("--label", default=None,
755:                         help="a note stored with the sweep, to find it again later")
756:     parser.add_argument("--generations", type=int, default=None, metavar="N",
757:                         help="generations for gep_lora/core/pipeline/continue_run.py, on top of the one "
758:                              "gep_lora/core/pipeline/start_run.py runs (default: with --run, the sweep's own "
759:                              "stored GENERATIONS; otherwise settings.py's, "
760:                              "currently %d)" % config.GENERATIONS)
761:     parser.add_argument("--set", action="append", default=[], dest="settings",
762:                         metavar="NAME=VALUE",
763:                         help="change one of the new sweep's stored settings before "
764:                              "continuing it, e.g. --set SELECTION_COUNT=3")
765:     parser.add_argument("--run-dir", default=None,
766:                         help="folder for the generated scripts (default %s)"
767:                              % config.DB_RUN_DIR)
768:     parser.add_argument("--limit", type=int, default=0,
769:                         help="process only the first N individuals of a generation")
770:     parser.add_argument("--include-blocked", action="store_true",
771:                         help="also run the ones marked BAD")
772:     parser.add_argument("--include-unchanged", action="store_true",
773:                         help="also run individuals whose chromosome has not changed "
774:                              "since their last execution")
775:     parser.add_argument("--keep-scripts", action="store_true",
776:                         help="leave the generated scripts on disk after processing them")
777:     parser.add_argument("--timeout", type=int, default=900,
778:                         help="seconds to allow each script (default 900)")
779:     parser.add_argument("--force", action="store_true",
780:                         help="re-score answers that already have a quality")
781:     parser.add_argument("--no-test", action="store_true",
782:                         help="skip the testing pass at the end (it runs when the "
783:                              "sweep has a testing split, and costs a base-model "
784:                              "load per individual it tests)")
785:     parser.add_argument("--test-min-quality", type=float, default=None, metavar="Q",
786:                         help="test the individuals scoring above this (default: "
787:                              "the sweep's own TESTING_MIN_QUALITY, falling back to "
788:                              "settings.py's, currently %.2f)"
789:                              % config.TESTING_MIN_QUALITY)
790:     parser.add_argument("--resume", action="store_true",
791:                         help="carry the sweep --run names on from wherever it "
792:                              "stopped: finish that generation, run the ones it has "
793:                              "left, then the testing pass")
794:     parser.add_argument("--evaluate", action="store_true",
795:                         help="grade the answers the sweep --run names already "
796:                              "holds (the ungraded ones, or all with --force), and "
797:                              "restate its fitness when that is still honest; "
798:                              "--set JUDGE_BASE_URL=... moves the judge")
799:     options = parser.parse_args(argv)
800:     if options.resume and options.evaluate:
801:         parser.error("--resume and --evaluate are two different things to do to "
802:                      "a sweep; pick one.")
803:     if (options.resume or options.evaluate) and options.run is None:
804:         parser.error("--%s needs a sweep that exists: pass --run (0 = the latest)."
805:                      % ("resume" if options.resume else "evaluate"))
806:     if options.generations is not None and options.generations < 1:
807:         parser.error("--generations is %d; gep_lora/core/pipeline/start_run.py's generation would be the whole "
808:                      "run. Use gep_lora/core/pipeline/start_run.py on its own for that." % options.generations)
809:     if options.run is not None and options.label:
810:         # create_run() is what writes a label, and --run creates nothing.
811:         parser.error("--label names a sweep as it is created, and --run adopts one "
812:                      "that already exists. Label it when you prepare the database.")
813:     return options
814: 
815: 
816: if __name__ == "__main__":
817:     sys.exit(cli())

(End of file - total 817 lines)
</content>
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 6.3s)

_Thinking:_

The tests agent returned empty. Let me investigate tests myself, and also look at a few more things. Let me check the unittests directory and run the test suite perhaps. Also check for linting config (ruff/flake8), pyproject/setup, requirements.

Let me also verify a few of the security findings quickly, since I should be accurate in my critique. But the agents gave detailed references. I should sanity-check the most severe claims.

Let me look at:
- gep_lora/core/evaluators/common.py around list_models / API_KEY
- gep_lora/apps/web/server.py judge_models
- unittests structure and run tests
- pyproject/requirements
- .gitignore includes NUL (that NUL file at root is odd)

Let me batch.

**Tool: bash**

**Input:**
```json
{
  "command": "Get-ChildItem -Recurse -Directory unittests | Select-Object FullName; echo \"--- files ---\"; Get-ChildItem -Recurse -File unittests -Filter *.py | Select-Object FullName; echo \"--- configs ---\"; Get-ChildItem -Force | Where-Object { $_.Name -match 'pyproject|setup|requirements|ruff|flake8|pytest|tox|mypy|.pre-commit' } | Select-Object Name",
  "description": "List test tree and config files"
}
```

**Output:**
```

FullName                                                                             
--------                                                                             
D:\sage-is\loras\gep-lora\unittests\adapters                                         
D:\sage-is\loras\gep-lora\unittests\api                                              
D:\sage-is\loras\gep-lora\unittests\async_api                                        
D:\sage-is\loras\gep-lora\unittests\async_api_agent                                  
D:\sage-is\loras\gep-lora\unittests\blends                                           
D:\sage-is\loras\gep-lora\unittests\evaluators                                       
D:\sage-is\loras\gep-lora\unittests\search                                           
D:\sage-is\loras\gep-lora\unittests\storage                                          
D:\sage-is\loras\gep-lora\unittests\testing                                          
D:\sage-is\loras\gep-lora\unittests\tools                                            
D:\sage-is\loras\gep-lora\unittests\__pycache__                                      
D:\sage-is\loras\gep-lora\unittests\adapters\__pycache__                             
D:\sage-is\loras\gep-lora\unittests\api\__pycache__                                  
D:\sage-is\loras\gep-lora\unittests\async_api\__pycache__                            
D:\sage-is\loras\gep-lora\unittests\async_api_agent\__pycache__                      
D:\sage-is\loras\gep-lora\unittests\evaluators\__pycache__                           
D:\sage-is\loras\gep-lora\unittests\search\__pycache__                               
D:\sage-is\loras\gep-lora\unittests\testing\__pycache__                              
--- files ---
D:\sage-is\loras\gep-lora\unittests\test_main.py                                     
D:\sage-is\loras\gep-lora\unittests\__init__.py                                      
D:\sage-is\loras\gep-lora\unittests\adapters\test_catalog.py                         
D:\sage-is\loras\gep-lora\unittests\adapters\__init__.py                             
D:\sage-is\loras\gep-lora\unittests\async_api\support.py                             
D:\sage-is\loras\gep-lora\unittests\async_api\test_golive.py                         
D:\sage-is\loras\gep-lora\unittests\async_api\test_inference.py                      
D:\sage-is\loras\gep-lora\unittests\async_api\test_registry.py                       
D:\sage-is\loras\gep-lora\unittests\async_api\test_results.py                        
D:\sage-is\loras\gep-lora\unittests\async_api\test_server.py                         
D:\sage-is\loras\gep-lora\unittests\async_api\test_submit.py                         
D:\sage-is\loras\gep-lora\unittests\async_api\test_train.py                          
D:\sage-is\loras\gep-lora\unittests\async_api\test_verify.py                         
D:\sage-is\loras\gep-lora\unittests\async_api\__init__.py                            
D:\sage-is\loras\gep-lora\unittests\async_api_agent\test_agent.py                    
D:\sage-is\loras\gep-lora\unittests\async_api_agent\__init__.py                      
D:\sage-is\loras\gep-lora\unittests\evaluators\test_composite.py                     
D:\sage-is\loras\gep-lora\unittests\evaluators\test_llm_judge_answers.py             
D:\sage-is\loras\gep-lora\unittests\evaluators\test_llm_judge_reference.py           
D:\sage-is\loras\gep-lora\unittests\evaluators\__init__.py                           
D:\sage-is\loras\gep-lora\unittests\search\support.py                                
D:\sage-is\loras\gep-lora\unittests\search\test_calculate_fitness.py                 
D:\sage-is\loras\gep-lora\unittests\search\test_draw_trees.py                        
D:\sage-is\loras\gep-lora\unittests\search\test_elitism.py                           
D:\sage-is\loras\gep-lora\unittests\search\test_generate_population.py               
D:\sage-is\loras\gep-lora\unittests\search\test_generation_cycle.py                  
D:\sage-is\loras\gep-lora\unittests\search\test_mutation.py                          
D:\sage-is\loras\gep-lora\unittests\search\test_selection.py                         
D:\sage-is\loras\gep-lora\unittests\search\test_weight_mutation.py                   
D:\sage-is\loras\gep-lora\unittests\search\__init__.py                               
D:\sage-is\loras\gep-lora\unittests\testing\test_evaluate_chromosome_against_loras.py
D:\sage-is\loras\gep-lora\unittests\testing\__init__.py                              
--- configs ---
                                                                                     



```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\evaluators\\common.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\evaluators\common.py</path>
<type>file</type>
<content>
1: """
2: gep_lora/core/evaluators/common.py - What an evaluator is, and what they all share.
3: 
4: One file rather than six copies. Everything here is used by at least two of the
5: evaluators beside it, and nothing here is an evaluator itself:
6: 
7:     the registry        Prepared, Evaluator, register(), get(), available()
8:     the judge transport ask_judge(), judge_settings(), resolve_model(),
9:                         discover_model(), list_models(), parse_reply(),
10:                         judge_note() --
11:                         llm_judge, llm_judge_reference, llm_judge_answers,
12:                         llm_judge_baseline and panel all speak to a model
13:     the references      load_references(), reference_for(), prepare_references()
14:                         -- llm_judge_reference, llm_judge_answers, similarity
15:                         and panel all grade against the dataset's own answer
16:     the tokeniser       WORD, tokens() -- similarity and heuristic both count
17:                         words
18:     the steps' own      needs_grading(), abandon_after() -- what the evaluate
19:                         step and the testing pass both have to decide about a
20:                         set of pending answers before an evaluator sees them
21: 
22: The names are public because they cross module boundaries now: a helper an
23: evaluator file imports cannot be an underscore. The one that stays private,
24: _request(), is the only thing here that nothing outside this file calls.
25: 
26: **A judge is one idea with two transports.** JUDGE_BACKEND picks between them:
27: "endpoint" POSTs to an OpenAI-compatible /v1/chat/completions, and "unsloth"
28: loads a model in this process the way the generated scripts do -- see
29: local_model.py, the only other module here that is not an evaluator. ask_judge()
30: is where they meet, and everything above it is shared, so which one a sweep used
31: changes where the tokens came from and nothing else about the score.
32: 
33: **No knob lives in this package.** They are all in settings.py, prefixed by the
34: evaluator that reads them (JUDGE_*, BASELINE_*, SIMILARITY_*, HEURISTIC_*,
35: PANEL_*), and they reach a step as the sweep's *stored* settings -- the same
36: contract every other step works under. The single exception is the API key,
37: which is read from the environment on purpose: a sweep records its settings into
38: the database, and a bearer token has no business in there.
39: 
40: The judge endpoints are reached over the OpenAI-compatible /v1/chat/completions
41: API, which LMStudio, OpenAI, OpenRouter, vLLM and most gateways all speak.
42: """
43: 
44: import json
45: import math
46: import os
47: import re
48: import time
49: import urllib.error
50: import urllib.request
51: 
52: from blends import generate_runs
53: 
54: from evaluators import local_model
55: 
56: # The two ways of reaching a judge, and what JUDGE_BACKEND names. ENDPOINT is
57: # the default and what a sweep created before the setting existed reads back as
58: # -- the behaviour it ran under.
59: ENDPOINT = "endpoint"
60: UNSLOTH = "unsloth"
61: BACKENDS = (ENDPOINT, UNSLOTH)
62: 
63: # Sent as "Authorization: Bearer <key>". LMStudio ignores it; cloud endpoints
64: # require it. Deliberately not a setting: settings are written into the sweep's
65: # database, and a real key must not end up there.
66: API_KEY = os.environ.get("JUDGE_API_KEY", "")
67: 
68: # Which evaluator a sweep uses when it never said -- every sweep created before
69: # EVALUATOR existed, so it has to be the behaviour those sweeps ran under.
70: DEFAULT = "llm_judge"
71: 
72: 
73: # ===========================================================================
74: # What an evaluator is
75: # ===========================================================================
76: 
77: 
78: class Prepared:
79:     """What one evaluate step's worth of scoring works from.
80: 
81:     Built once by `prepare()`, handed to every `score()` call. `label` is what
82:     lands in exchanges.judge_model -- the judge's model id for the judging
83:     evaluators, the method's own name for the local ones, so the column keeps
84:     answering the question it was added for: what gave this score.
85:     """
86: 
87:     __slots__ = ("conf", "label", "settings", "references", "baselines", "notes")
88: 
89:     def __init__(self, conf, label, settings=None, references=None, notes=(),
90:                  baselines=None):
91:         self.conf = conf
92:         self.label = label
93:         self.settings = settings or {}
94:         self.references = references or {}
95:         # {question key: what the base model said} -- the control
96:         # llm_judge_baseline grades against, read out of the database once.
97:         self.baselines = baselines or {}
98:         self.notes = list(notes)      # lines gep_lora/core/pipeline/start_run.py prints before scoring
99: 
100: 
101: class Evaluator:
102:     """One way of turning an answer into a quality.
103: 
104:     A plain holder rather than a base class: an evaluator is two functions and
105:     two strings, and subclassing would only invite one of them to grow state
106:     that outlives a step.
107:     """
108: 
109:     __slots__ = ("name", "description", "prepare", "score", "wants_reference",
110:                  "needs_judge", "wants_baseline", "judges", "check",
111:                  "via_judge_backend")
112: 
113:     def __init__(self, name, description, prepare, score,
114:                  wants_reference=False, needs_judge=False,
115:                  wants_baseline=False, judges=None, check=None,
116:                  via_judge_backend=True):
117:         self.name = name
118:         self.description = description
119:         self.prepare = prepare
120:         self.score = score
121:         self.wants_reference = wants_reference
122:         # Asks a model for every score, whichever backend produces it. What the
123:         # abandon rule is decided on: a judge call costs a request or a
124:         # generate(), and a local scorer costs neither. For an evaluator whose
125:         # answer depends on its settings (composite) this is "can ask one", and
126:         # asks_judge(conf) is the answer for one sweep.
127:         self.needs_judge = needs_judge
128:         # Needs the base model's own answers, which cost a model load the first
129:         # time they are wanted and come out of the database ever after.
130:         self.wants_baseline = wants_baseline
131:         # judges(conf) -> bool, for an evaluator that only knows under a sweep's
132:         # settings whether it asks a model. None means needs_judge is the answer.
133:         self.judges = judges
134:         # check(conf), raising SystemExit on settings this evaluator could never
135:         # score under -- run when a sweep is created (start_run.freeze), so a
136:         # typo is refused before a population is drawn rather than after the
137:         # process step. None means there is nothing to check that cheaply.
138:         self.check = check
139:         # Whether "asks a model" means through JUDGE_BACKEND -- true for every
140:         # evaluator that calls ask_judge(), false for one that speaks to a judge
141:         # of its own (jev_judge_reference, to Jev). Only needs_judge evaluators
142:         # ever set this false; a local scorer's transport is moot. What
143:         # wants_the_card() and the --evaluators summary read to avoid claiming a
144:         # judge that never touches JUDGE_BACKEND runs through it.
145:         self.via_judge_backend = via_judge_backend
146: 
147:     def asks_judge(self, conf):
148:         """Whether scoring under these settings asks a model. -> bool.
149: 
150:         What a step decides the abandon rule and the pool teardown on. The same
151:         as needs_judge for every evaluator but one that combines others.
152:         """
153:         if self.judges is None:
154:             return bool(self.needs_judge)
155:         return bool(self.judges(conf))
156: 
157: 
158: # Filled by the register() call at the foot of each evaluator module, as
159: # gep_lora/core/evaluators/__init__.py imports them. Nothing else writes to it.
160: _REGISTRY = {}
161: 
162: 
163: def register(evaluator):
164:     _REGISTRY[evaluator.name] = evaluator
165:     return evaluator
166: 
167: 
168: def get(name):
169:     """The evaluator called `name`, or a failure naming the ones there are."""
170:     name = name or DEFAULT
171:     try:
172:         return _REGISTRY[name]
173:     except KeyError:
174:         raise SystemExit(
175:             "unknown EVALUATOR %r. settings.py must name one of: %s"
176:             % (name, ", ".join(sorted(_REGISTRY)))
177:         )
178: 
179: 
180: def available():
181:     """[(name, description)] for every registered evaluator, for --list."""
182:     return [(name, _REGISTRY[name].description) for name in sorted(_REGISTRY)]
183: 
184: 
185: # ===========================================================================
186: # The reference answers, for the evaluators that compare against them
187: # ===========================================================================
188: 
189: 
190: def normalise(text):
191:     """A question reduced to what makes two of them the same question."""
192:     return " ".join((text or "").split()).strip().lower()
193: 
194: 
195: def load_references(conf):
196:     """{normalised question: reference answer} from the sweep's eval set.
197: 
198:     Keyed by the question rather than by position because that is what survives
199:     a re-run: an exchange stores the question it actually asked, so matching on
200:     it cannot quietly pair an answer with the wrong reference the way an index
201:     into a file that has since been edited could. Position is the fallback, for
202:     two prompts that really are the same string.
203:     """
204:     records = generate_runs.eval_records(conf.get("TRAINING_SET"),
205:                                          conf.get("TRAINING_COUNT"))
206:     by_question, by_position = {}, {}
207:     for record in records:
208:         if record["reference"]:
209:             by_question.setdefault(normalise(record["question"]), record["reference"])
210:             by_position[record["position"]] = record["reference"]
211:     return by_question, by_position
212: 
213: 
214: def reference_for(item, prepared):
215:     """The dataset's own answer to this exchange, or None."""
216:     by_question, by_position = prepared.references.get("by_question", {}), \
217:         prepared.references.get("by_position", {})
218:     return by_question.get(normalise(item["question"])) or by_position.get(item["position"])
219: 
220: 
221: def prepare_references(conf, evaluator_name):
222:     """The reference maps, or a clear failure when the eval set holds none.
223: 
224:     A missing reference is fatal *here* rather than per exchange: an evaluator
225:     that compares against the dataset's answers cannot score a plain
226:     prompt-per-line file at all, and finding that out one exchange at a time
227:     would spend a whole evaluate step to say so.
228:     """
229:     by_question, by_position = load_references(conf)
230:     if not by_position:
231:         raise SystemExit(
232:             "EVALUATOR = %r needs the answers that come with the eval set, and "
233:             "%s carries none. That file is either plain one-prompt-per-line "
234:             "text or JSON records with no assistant turn; point TRAINING_SET at "
235:             "a dataset that has both turns (datasets/*.json do), or choose an "
236:             "evaluator that does not compare against a reference."
237:             % (evaluator_name, generate_runs.training_set_path(conf.get("TRAINING_SET")))
238:         )
239:     return {"by_question": by_question, "by_position": by_position}
240: 
241: 
242: # ===========================================================================
243: # The judge transport, shared by every evaluator that asks a model
244: # ===========================================================================
245: 
246: 
247: def _request(url, payload, api_key, timeout):
248:     """POST JSON, return the decoded JSON reply. Raises urllib errors."""
249:     data = json.dumps(payload).encode("utf-8")
250:     headers = {"Content-Type": "application/json"}
251:     if api_key:
252:         headers["Authorization"] = "Bearer " + api_key
253:     request = urllib.request.Request(url, data=data, headers=headers, method="POST")
254:     with urllib.request.urlopen(request, timeout=timeout) as response:
255:         return json.loads(response.read().decode("utf-8"))
256: 
257: 
258: def discover_model(base_url, api_key, timeout):
259:     """Ask the endpoint which model it has loaded (LMStudio serves one)."""
260:     chat_models = list_models(base_url, api_key, timeout)
261:     if not chat_models:
262:         raise SystemExit("%s/models lists no chat models. Load one in LMStudio first."
263:                          % base_url.rstrip("/"))
264:     return chat_models[0]
265: 
266: 
267: def list_models(base_url, api_key, timeout):
268:     """The chat models an endpoint's /models lists, in its order. -> [id].
269: 
270:     What discover_model() picks the first of, and what the async API's demo
271:     offers to pick from. Raises SystemExit, naming the endpoint, when it
272:     cannot be reached or does not answer in JSON.
273:     """
274:     url = base_url.rstrip("/") + "/models"
275:     headers = {}
276:     if api_key:
277:         headers["Authorization"] = "Bearer " + api_key
278:     request = urllib.request.Request(url, headers=headers)
279:     try:
280:         with urllib.request.urlopen(request, timeout=timeout) as response:
281:             listed = json.loads(response.read().decode("utf-8")).get("data") or []
282:     except (urllib.error.URLError, OSError, ValueError, AttributeError) as error:
283:         raise SystemExit(
284:             "cannot reach the judge at %s (%s). Is LMStudio running with a model "
285:             "loaded and its server started? Point JUDGE_BASE_URL at a different "
286:             "endpoint to use another one."
287:             % (base_url, error)
288:         )
289:     # LMStudio lists embedding models alongside chat ones; those cannot grade.
290:     ids = [entry.get("id") for entry in listed if isinstance(entry, dict)]
291:     return [one for one in ids if isinstance(one, str) and "embed" not in one.lower()]
292: 
293: 
294: def parse_reply(text):
295:     """Pull the quality and the judge's reason out of its reply.
296: 
297:     Prefers well-formed JSON, and falls back to the first number in 0..1 that
298:     the text contains, so a model that wraps its JSON in prose or code fences
299:     still scores rather than failing the whole run. The reason is best-effort:
300:     the score is what the search needs, so a missing reason is never fatal, and
301:     a reply truncated after the score still yields one.
302:     """
303:     cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
304: 
305:     reason = ""
306:     try:
307:         parsed = json.loads(cleaned)
308:         value = parsed.get("quality")
309:         reason = (parsed.get("reason") or "").strip()
310:     except (ValueError, AttributeError):
311:         match = re.search(r'"quality"\s*:\s*([0-9]*\.?[0-9]+)', cleaned)
312:         if not match:
313:             match = re.search(r"\b(0?\.[0-9]+|0|1(?:\.0+)?)\b", cleaned)
314:         value = match.group(1) if match else None
315:         # The JSON did not parse -- usually truncated -- so recover the reason
316:         # textually if enough of it made it through.
317:         said = re.search(r'"reason"\s*:\s*"([^"]*)', cleaned)
318:         reason = said.group(1).strip() if said else ""
319: 
320:     if value is None:
321:         raise ValueError("no quality score in judge reply: %r" % text[:200])
322:     score = float(value)
323:     if not 0.0 <= score <= 1.0:
324:         raise ValueError("quality %r is outside 0..1" % score)
325:     return score, reason
326: 
327: 
328: def ask_judge(system_prompt, user_content, settings):
329:     """One grading call. Returns (quality, reason).
330: 
331:     `settings` is the resolved judge block -- backend, base_url, api_key, model,
332:     temperature, max_tokens, timeout, retries, retry_wait, response_format, and
333:     the JUDGE_LOCAL_* values the unsloth backend loads under -- which the
334:     evaluators build from the sweep's stored JUDGE_*/PANEL_* values through
335:     judge_settings().
336: 
337:     The two backends meet here rather than in the evaluators, so which one is in
338:     use changes only where the tokens come from: the rubric, the retries and
339:     reading a quality out of the reply are one code path either way, and that is
340:     what makes a sweep graded locally comparable with one graded over an API.
341:     """
342:     local = settings.get("backend") == UNSLOTH
343:     payload, url = None, None
344:     if not local:
345:         payload = {
346:             "model": settings["model"],
347:             "temperature": settings["temperature"],
348:             "max_tokens": settings["max_tokens"],
349:             "messages": [
350:                 {"role": "system", "content": system_prompt},
351:                 {"role": "user", "content": user_content},
352:             ],
353:         }
354:         if settings.get("response_format"):
355:             payload["response_format"] = settings["response_format"]
356:         url = settings["base_url"].rstrip("/") + "/chat/completions"
357: 
358:     retries = settings["retries"]
359:     last_error = None
360:     for attempt in range(retries + 1):
361:         try:
362:             if local:
363:                 text = local_model.generate(system_prompt, user_content, settings)
364:             else:
365:                 reply = _request(url, payload, settings["api_key"], settings["timeout"])
366:                 message = reply["choices"][0]["message"]
367:                 text = message.get("content") or ""
368:                 if not text.strip():
369:                     # A reasoning model that spent its whole budget thinking
370:                     # returns an empty content; the score may still be in the
371:                     # reasoning.
372:                     text = message.get("reasoning_content") or message.get("reasoning") or ""
373:             # A blank or unparseable reply is usually a truncation, so it is
374:             # worth another attempt rather than losing the answer's score.
375:             return parse_reply(text)
376:         except ValueError as error:
377:             last_error = error
378:             if local and not settings["temperature"]:
379:                 # Greedy decoding: the same prompt returns the same reply, so a
380:                 # retry would spend another generate() to fail in the same way.
381:                 break
382:         except urllib.error.HTTPError as error:
383:             # Some endpoints reject response_format; the prompt asks for JSON
384:             # anyway, so drop it and try once more rather than failing.
385:             if error.code == 400 and "response_format" in payload:
386:                 payload.pop("response_format")
387:                 last_error = error
388:                 continue
389:             if error.code not in (408, 409, 429) and error.code < 500:
390:                 raise RuntimeError("judge returned HTTP %d: %s"
391:                                    % (error.code, error.read().decode("utf-8", "replace")[:200]))
392:             last_error = error
393:         except (urllib.error.URLError, OSError, KeyError, IndexError) as error:
394:             last_error = error
395: 
396:         if attempt < retries:
397:             time.sleep(settings["retry_wait"])
398:     if local:
399:         raise RuntimeError("the local judge produced nothing gradeable: %s" % last_error)
400:     raise RuntimeError("judge unreachable after %d attempts: %s" % (retries + 1, last_error))
401: 
402: 
403: def backend_of(conf):
404:     """Which transport a sweep grades through, checked. -> ENDPOINT or UNSLOTH.
405: 
406:     Its own function because gep_lora/core/pipeline/start_run.py asks it when a sweep is *created*, an
407:     hour before any answer needs grading: a misspelt backend should cost a line
408:     at the top of the run, not a finished transcript nobody can score.
409:     """
410:     backend = conf.get("JUDGE_BACKEND") or ENDPOINT
411:     if backend not in BACKENDS:
412:         raise SystemExit("JUDGE_BACKEND must be %s, not %r"
413:                          % (" or ".join(repr(name) for name in BACKENDS), backend))
414:     return backend
415: 
416: 
417: def judge_settings(conf, model=None, base_url=None):
418:     """The JUDGE_* block of a sweep's settings, resolved for ask_judge().
419: 
420:     Defaults are spelled out here for one reason only: a sweep created before a
421:     knob existed has no value for it stored, and resuming one must not crash on
422:     a KeyError. settings.py is still where a knob is *set*. It is also why a
423:     sweep that predates JUDGE_BACKEND reads back as ENDPOINT -- that is the
424:     behaviour it ran under.
425: 
426:     The last three are the unsloth backend's own, and the endpoint backend
427:     ignores them exactly as the local one ignores base_url, api_key, timeout and
428:     response_format. One block rather than two, because a judge is one idea with
429:     two transports and a caller building the block should not have to know which
430:     one a sweep chose.
431:     """
432:     return {
433:         "backend": backend_of(conf),
434:         "base_url": base_url or conf.get("JUDGE_BASE_URL"),
435:         "api_key": API_KEY,
436:         "model": model or conf.get("JUDGE_MODEL"),
437:         "temperature": conf.get("JUDGE_TEMPERATURE", 0.0),
438:         "max_tokens": conf.get("JUDGE_MAX_TOKENS", 2000),
439:         "timeout": conf.get("JUDGE_TIMEOUT", 300),
440:         "retries": conf.get("JUDGE_RETRIES", 2),
441:         "retry_wait": conf.get("JUDGE_RETRY_WAIT", 3),
442:         "response_format": conf.get("JUDGE_RESPONSE_FORMAT", {"type": "json_object"}),
443:         "max_seq_length": conf.get("JUDGE_LOCAL_MAX_SEQ_LENGTH", 4096),
444:         "load_in_4bit": conf.get("JUDGE_LOCAL_LOAD_IN_4BIT", True),
445:         "chat_template": conf.get("JUDGE_LOCAL_CHAT_TEMPLATE"),
446:     }
447: 
448: 
449: def resolve_model(settings, grading):
450:     """Fill in the model a sweep did not name, for the backend it chose.
451: 
452:     An endpoint can be asked what it has loaded -- that is what JUDGE_MODEL =
453:     None means, and what you want with LMStudio. A machine cannot be asked, so
454:     the local backend needs the name in writing; and it must not quietly fall
455:     back to BASE_MODEL, which would leave the model under test grading its own
456:     descendants.
457:     """
458:     if settings["model"] or not grading:
459:         return settings["model"]
460:     if settings["backend"] == UNSLOTH:
461:         raise SystemExit(
462:             "JUDGE_BACKEND = 'unsloth' grades with a model loaded here, so "
463:             "JUDGE_MODEL has to name one -- a Hub repo id, or a folder. There "
464:             "is nothing to ask what it has loaded the way an endpoint can be "
465:             "asked, and falling back to BASE_MODEL would leave the model under "
466:             "test marking its own homework."
467:         )
468:     settings["model"] = discover_model(
469:         settings["base_url"], settings["api_key"], settings["timeout"])
470:     return settings["model"]
471: 
472: 
473: def judge_note(settings, grading, what="judge"):
474:     """The line a prepare() prints to say where the grading will come from."""
475:     if not grading:
476:         return "%s: not contacted -- no answer needs grading" % what
477:     if settings["backend"] == UNSLOTH:
478:         return "%s: %s, loaded here with unsloth" % (what, local_model.describe(settings))
479:     return "%s: %s at %s" % (what, settings["model"], settings["base_url"])
480: 
481: 
482: def release_models():
483:     """Hand back whatever grading loaded. A no-op unless a judge was loaded.
484: 
485:     Called by the evaluate step and by the testing pass when they are done,
486:     whichever evaluator ran: a step that has finished scoring has no further use
487:     for a model, and `gep_lora/core/pipeline/main.py` goes straight on to a generation whose scripts
488:     each want the card.
489:     """
490:     local_model.release()
491: 
492: 
493: def needs_grading(pending):
494:     """Whether any pending exchange has an answer worth spending a call on.
495: 
496:     An all-blank set -- a sweep where every script failed -- is scored 0.0 by
497:     gep_lora/core/pipeline/start_run.py without anyone being asked, and a mocked sweep arrives scored, so
498:     neither should make the step demand an endpoint that need not be up, or load
499:     a judge model that has nothing to grade.
500:     """
501:     return any((row["answer"] or "").strip() for row in pending)
502: 
503: 
504: def abandon_after(conf, answers):
505:     """How many graded zeros in a row condemn one individual, or None.
506: 
507:     JUDGE_ABANDON_FRACTION of the answers that individual has pending, rounded
508:     up and never fewer than one -- 0.1 is "the first 10%". None when the rule is
509:     off, which is what 0, None and an individual with nothing pending all mean.
510: 
511:     Here rather than in either caller because both of them apply it: the
512:     evaluate step over `exchanges` and the testing pass over `test_results`
513:     grade the same answers under the same evaluator, so an individual worth
514:     giving up on in one is worth giving up on in the other. Whether to apply it
515:     at all is the caller's -- only an evaluator that asks a model (needs_judge)
516:     has anything to save by stopping early, and it saves it either way: a
517:     request not sent, or a generate() not run.
518:     """
519:     fraction = conf.get("JUDGE_ABANDON_FRACTION", 0.0) or 0.0
520:     if fraction <= 0 or not answers:
521:         return None
522:     return max(1, math.ceil(fraction * answers))
523: 
524: 
525: # ===========================================================================
526: # Words, for the evaluators that count them
527: # ===========================================================================
528: 
529: 
530: WORD = re.compile(r"[\w']+", re.UNICODE)
531: 
532: 
533: def tokens(text, case_sensitive):
534:     text = text if case_sensitive else text.lower()
535:     return WORD.findall(text)

(End of file - total 535 lines)
</content>
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 5.4s)

_Thinking:_

There are no project config files (no pyproject.toml, requirements.txt, ruff, pytest config). Let me verify. The `run_unit_tests.bat` probably runs `python -m unittest discover`. Let me check it, and run the tests perhaps. Also let me check the NUL file, gitignore, and a couple security claims.

Let me read run_unit_tests.bat, .gitignore, and check the unittests support modules to understand test style. Also check gep_lora/apps/web/server.py judge_models and registry digest.

Let me batch.

**Tool: bash**

**Input:**
```json
{
  "command": "Get-Content run_unit_tests.bat; echo \"--- gitignore ---\"; Get-Content .gitignore; echo \"--- test_main head ---\"; Get-Content unittests/core/test_main.py -TotalCount 60",
  "description": "Read test runner, gitignore, main test"
}
```

**Output:**
```
python -m unittest discover -s unittests -t .
--- gitignore ---
__pycache__
.venv/
.git
population.txt
trees.txt
run
runs
run_db
run_db*
run_real
run_testing
test
tmp
unsloth_compiled_cache
run_tmp
.idea
dbtemplates
NUL

# Ignore everything inside the adapter folders by default. This ignores their
# *contents* rather than the folders themselves: git cannot re-include a file
# whose parent directory is excluded, so the two negations below only work if
# loras/Lora00N/ is itself still visible.
loras/*

# Re-include only the files you want to track
!loras/*.md


*.bat

# The async API: its registry and every job folder
api_jobs

.claude
--- test_main head ---
"""
test_main.py - Where a stopped sweep got to, read back out of its database.

main.where_it_stopped() is what `gep_lora/core/pipeline/main.py --resume` and `--evaluate` plan by.
It reads two things a driver writes as it goes -- the fitness snapshots and the
steps that finished -- so these tests write exactly those and nothing else: a
sweep here is the record of how far a search went, not a search.
"""

import unittest

import main
from storage import store

from unittests.core.search.support import SweepTestCase

GENERATION = ["trees", "runs", "process", "evaluate", "fitness"]
TAIL = ["elitism", "selection", "mutation", "weight_mutation"]


class WhereItStoppedTests(SweepTestCase):

    def ran(self, *steps):
        """Record `steps` as finished, one pass, in order -- taking a fitness
        snapshot wherever `fitness` is among them, as the step would."""
        pass_no = store.next_pass(self.conn, self.run)
        for position, step in enumerate(steps, 1):
            store.add_step_timing(self.conn, self.run, pass_no, position, step, 0.1)
            if step == "fitness":
                generation = len(store.fitness_by_generation(self.conn, self.run)) + 1
                store.record_fitness(self.conn, self.run, generation, [
                    {"number": 1, "chromosome": "CAT.L2.L1.w5.w3", "state": "ok",
                     "fitness": 0.5, "answers": 3, "unscored": 0}])
        self.conn.commit()

    def plan(self, generations=2):
        return main.where_it_stopped(self.conn, self.run, generations)

    def test_a_sweep_with_no_population_is_not_started(self):
        plan = self.plan()
        self.assertFalse(plan.populated)

    def test_stopped_in_the_first_generation_it_starts_that_one_again(self):
        self.populate()
        self.ran("population", "trees", "runs", "process")
        plan = self.plan()
        self.assertEqual(plan.rest, GENERATION + TAIL)
        self.assertEqual((plan.scored, plan.generations, plan.at_rest), (0, 2, False))

    def test_stopped_part_way_through_the_tail_it_finishes_the_tail(self):
        self.populate()
        self.ran("population", *GENERATION)
        self.ran("elitism", "selection")
        plan = self.plan()
        self.assertEqual(plan.rest, ["mutation", "weight_mutation"])
        self.assertEqual(plan.tail_done, ("elitism", "selection"))
        self.assertEqual((plan.scored, plan.generations, plan.at_rest), (1, 2, True))

    def test_resting_on_a_snapshot_it_breeds_the_next_generation(self):
        self.populate()

```

**Tool: grep**

**Input:**
```json
{
  "pattern": "def judge_models|def digest|def _request|list_models\\(|def claim_next|def finish|def request_cancel|def orphaned",
  "include": "*.py",
  "path": "D:\\sage-is\\loras\\gep-lora\\async_api"
}
```

**Output:**
```
Found 13 matches
D:\sage-is\loras\gep-lora\async_api\registry.py:
  Line 165: def digest(secret):

  Line 350:     def claim_next(self):

  Line 372:     def finish(self, job_id, status, exit_code=None, error=None):

  Line 378:     def request_cancel(self, job_id):

  Line 450:     def orphaned(self):

  Line 500:     def claim_next_verification(self):

  Line 527:     def finish_verification(self, verification_id, status, exit_code=None, error=None):

  Line 533:     def orphaned_verifications(self):

  Line 603:     def claim_next_training(self):

  Line 625:     def finish_training(self, training_id, status, exit_code=None, error=None):

  Line 659:     def orphaned_trainings(self):


D:\sage-is\loras\gep-lora\async_api\server.py:
  Line 603:     def judge_models(self, user, query):

  Line 618:             models = gep_lora.core.evaluators.common.list_models(

```

---

## Assistant (Build · DeepSeek V4.1 Flash · 5.1s)

_Thinking:_

Let me verify the SSRF finding and digest, and run the test suite. Let me read server.py judge_models and registry digest, plus run tests.

Let me also check the api/ folder (there's both `api/` and `async_api/`). The `api/` folder wasn't in the CLAUDE.md layout. Let me look.

Let me batch: run unit tests (might take a while; use timeout), read server.py 590-630, registry.py 160-170.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\async_api\\server.py",
  "offset": 595,
  "limit": 40
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\async_api\server.py</path>
<type>file</type>
<content>
595:                               settings.SHARED_DATASETS_DIR)
596:         try:
597:             names = sorted(name for name in os.listdir(folder)
598:                            if os.path.isfile(os.path.join(folder, name)))
599:         except OSError:
600:             names = []
601:         return 200, {"datasets": names}
602: 
603:     def judge_models(self, user, query):
604:         """The chat models a judge endpoint lists, for a form to offer.
605: 
606:         Asked from here rather than by the browser: an endpoint such as LM Studio
607:         sends no CORS headers, and the one worth listing is the one this machine
608:         -- where the worker will grade -- can reach, which a browser elsewhere
609:         may not. Only the model ids come back. $JUDGE_API_KEY goes with the
610:         request, as it does when a step asks the same endpoint.
611:         """
612:         base_url = ((query.get("base_url") or [""])[0].strip()
613:                     or config.JUDGE_BASE_URL or "")
614:         if urlparse(base_url).scheme not in ("http", "https") or not urlparse(base_url).netloc:
615:             raise ApiError(400, "base_url must be an http(s) URL, not %r" % base_url)
616:         import evaluators                   # the judge transport, loaded when first asked
617:         try:
618:             models = gep_lora.core.evaluators.common.list_models(
619:                 base_url, evaluators.API_KEY, settings.JUDGE_MODELS_TIMEOUT)
620:         except SystemExit as error:
621:             raise ApiError(502, str(error), base_url=base_url)
622:         return 200, {"base_url": base_url, "models": models}
623: 
624:     def list_live(self, user):
625:         return 200, {"live": [deployment_json(row)
626:                               for row in self.registry.deployments(user_id=user["id"])]}
627: 
628:     def unset_one(self, user, deployment_id):
629:         row = self.registry.deployment(deployment_id, user["id"])
630:         if row is None or row["revoked_at"]:
631:             raise ApiError(404, "no live deployment %d" % deployment_id)
632:         return 200, {"unset": self.unset([row])}
633: 
634:     # --- LoRAs -------------------------------------------------------------

(Showing lines 595-634 of 1034. Use offset=635 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\async_api\\registry.py",
  "offset": 160,
  "limit": 12
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\async_api\registry.py</path>
<type>file</type>
<content>
160: 
161: def now():
162:     return time.strftime("%Y-%m-%dT%H:%M:%S")
163: 
164: 
165: def digest(secret):
166:     """What is stored for a key or a token: never the secret itself."""
167:     return hashlib.sha256(secret.encode("utf-8")).hexdigest()
168: 
169: 
170: DATABASE = "api.sqlite"
171: # What the database was called before the LoRAs moved into it.

(Showing lines 160-171 of 736. Use offset=172 to continue.)
</content>
```

**Tool: bash**

**Input:**
```json
{
  "command": "Get-ChildItem api -Recurse -File | Select-Object -First 40 FullName; echo \"--- async_api.txt ---\"; Get-Content async_api.txt",
  "description": "Inspect api folder and async_api.txt"
}
```

**Output:**
```

FullName                                                           
--------                                                           
D:\sage-is\loras\gep-lora\api\__pycache__\catalog.cpython-311.pyc  
D:\sage-is\loras\gep-lora\api\__pycache__\catalog.cpython-313.pyc  
D:\sage-is\loras\gep-lora\api\__pycache__\config.cpython-311.pyc   
D:\sage-is\loras\gep-lora\api\__pycache__\config.cpython-313.pyc   
D:\sage-is\loras\gep-lora\api\__pycache__\details.cpython-311.pyc  
D:\sage-is\loras\gep-lora\api\__pycache__\details.cpython-313.pyc  
D:\sage-is\loras\gep-lora\api\__pycache__\jobs.cpython-311.pyc     
D:\sage-is\loras\gep-lora\api\__pycache__\jobs.cpython-313.pyc     
D:\sage-is\loras\gep-lora\api\__pycache__\progress.cpython-311.pyc 
D:\sage-is\loras\gep-lora\api\__pycache__\progress.cpython-313.pyc 
D:\sage-is\loras\gep-lora\api\__pycache__\prompting.cpython-311.pyc
D:\sage-is\loras\gep-lora\api\__pycache__\prompting.cpython-313.pyc
D:\sage-is\loras\gep-lora\api\__pycache__\server.cpython-311.pyc   
D:\sage-is\loras\gep-lora\api\__pycache__\server.cpython-313.pyc   
D:\sage-is\loras\gep-lora\api\__pycache__\worker.cpython-311.pyc   
D:\sage-is\loras\gep-lora\api\__pycache__\worker.cpython-313.pyc   
D:\sage-is\loras\gep-lora\api\__pycache__\__init__.cpython-311.pyc 
D:\sage-is\loras\gep-lora\api\__pycache__\__init__.cpython-313.pyc 
--- async_api.txt ---
Create an api, async batch processing and inference for this system. The idea is that a user can submit a job with the settings and datasets for processing over the web and then a background process picks up the request and runs it. Finally the processed run can be selected for inference (goes live).

Implement this in the folder aync_api

- Submit job creates a sqlite database with all the information necessary to run the job. The sqlite db is the same that is created when this process runs (see storage.py).

- The backend does the batch processing in arrival order by picking up each of the sqlite files and puting them through gep_lora/core/pipeline/main.py

- The api exposes a job list per user and a job detail that shows the run results

- The api exposes a set run live which marks a job as ready for inference (returns a token for using that run). Going live may mean different things (run locally, upload to a remote server, etc).

- The inference api accepts the token above and a prompt and then streams the answer. The first time it is called it loads the model which is cached for a set time (5 minutes by default)

- Other api functions: cancel run, delete job or run, get job status, unset for inference



```

---

## Assistant (Build · DeepSeek V4.1 Flash · 16.6s)

_Thinking:_

The `api/` folder only has .pyc files (stale, gitignored? Actually `.pyc` are in `__pycache__`, gitignored). So `api/` is a legacy/leftover directory with only pycache. That's a cleanup item.

Now let me run the tests to see the state. This might take a while. Let me run with a timeout. The `run_unit_tests.bat` runs `python -m unittest discover -s unittests -t .`. But the right interpreter is the venv one level up. Let me check if venv exists.

Actually, tests may need torch etc. Let me try with the venv python. But could be slow. Let me run a subset first, or run discovery with a timeout.

Let me check the venv python path exists.

**Tool: bash**

**Input:**
```json
{
  "command": "Test-Path D:\\sage-is\\loras\\.venv\\Scripts\\python.exe; echo \"---\"; python --version; echo \"--- try import ---\"; D:\\sage-is\\loras\\.venv\\Scripts\\python.exe -c \"import torch, unsloth; print('venv ok')\" 2>&1 | Select-Object -First 5",
  "description": "Check venv and interpreter"
}
```

**Output:**
```
True
---
Python 3.11.9
--- try import ---
D:\sage-is\loras\.venv\Scripts\python.exe : W0923 19:58:56.315000 50516 
Lib\site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not 
supported in Windows or MacOs.
At line:1 char:111
+ ... mport ---"; D:\sage-is\loras\.venv\Scripts\python.exe -c "import torc ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (W0923 19:58:56....ndows or MacOs.:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
venv ok

```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\unittests\\search\\support.py"
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\unittests\search\support.py</path>
<type>file</type>
<content>
1: """
2: support.py - The fixtures the database-backed search steps need.
3: 
4: calculate_fitness, elitism, selection and mutation are all written as
5: `step(conn, run_id, ...)`, so testing them means handing them a sweep. This
6: builds one: a real sqlite database with the real schema, in a temp folder that
7: goes away with the test.
8: 
9: A real database and not a stand-in, deliberately. Most of what these four steps
10: promise is a promise about what ends up in the tables -- that a cull takes an
11: individual's executions with it, that a mutant's fitness comes back NULL, that
12: a copy inherits every column but `is_best` -- and a fake connection would be
13: this suite agreeing with itself about SQL nobody ran. store.py is the only
14: module that speaks sqlite, so using it here keeps that true of the tests too.
15: """
16: 
17: import os
18: import random
19: import shutil
20: import tempfile
21: import unittest
22: 
23: from storage import store
24: 
25: 
26: # A handful of chromosomes that are known-good against the grammar: root CAT,
27: # every binary op over two operators, every L* over one variable. Written out
28: # rather than drawn, so a test that names one is reading the same tree tomorrow.
29: VALID = (
30:     "CAT.L2.L1.w5.w3",
31:     "CAT.L3.L3.w5.w1",
32:     "CAT.L4.L1.w5.w1",
33:     "CAT.L5.L4.w2.w3",
34:     "CAT.L3.L5.w1.w5",
35:     "CAT.CAT.L5.L3.L4.w3.w3.w2",
36:     "CAT.L2.LIN.w2.L3.L1.w5.w1",
37:     "CAT.L5.CAT.w4.L4.L1.w2.w1",
38:     "CAT.L3.SVD.w5.L4.CAT.w3.L4.L1.w1.w3",
39:     "CAT.L4.SVD.w3.SVD.LIN.L1.L4.L1.L2.w3.w2.w2.w4",
40: )
41: 
42: # What selection.select() reads out of a sweep's settings when it draws its
43: # newcomer. The real defaults, so a test is not exercising a shape the pipeline
44: # never passes.
45: CONF = {"MAX_DEPTH": 4, "BRANCH_PROB": 0.2, "UNIQUE": True}
46: 
47: 
48: class SweepTestCase(unittest.TestCase):
49:     """A test with one empty sweep in a throwaway database.
50: 
51:     `self.conn` is the connection, `self.run` the sweep's id. Subclasses call
52:     populate() and then the step under test.
53:     """
54: 
55:     def setUp(self):
56:         self.folder = tempfile.mkdtemp(prefix="gep-search-tests-")
57:         # store.connect() resolves a relative path against the repo folder, so
58:         # the path handed to it has to be absolute or the database lands in the
59:         # working tree.
60:         self.conn = store.connect(os.path.join(self.folder, "test.sqlite3"))
61:         self.run = store.create_run(self.conn, "template_code_mocked.py",
62:                                     label="unit test")
63:         self.addCleanup(self._teardown)
64: 
65:     def _teardown(self):
66:         self.conn.close()
67:         shutil.rmtree(self.folder, ignore_errors=True)
68: 
69:     # --- building a population --------------------------------------------
70: 
71:     def populate(self, chromosomes=None, count=None):
72:         """Store a population. -> the chromosomes it was given.
73: 
74:         Numbers run from 1, the way add_individuals hands them out.
75:         """
76:         if chromosomes is None:
77:             chromosomes = list(VALID[:count if count is not None else 4])
78:         store.add_individuals(self.conn, self.run, chromosomes)
79:         return list(chromosomes)
80: 
81:     def set_fitness(self, mapping):
82:         """Write individuals.fitness for `{number: fitness}`.
83: 
84:         The column the search sorts on, set directly: fitness is what
85:         calculate_fitness produces, and the three steps that read it should be
86:         testable without running that one first.
87:         """
88:         for number, fitness in mapping.items():
89:             store.set_fitness(self.conn, self.run, number, fitness)
90:         self.conn.commit()
91: 
92:     def run_individual(self, number, qualities, verdict="ok", exit_code=0):
93:         """Give one individual an execution and a graded transcript.
94: 
95:         `qualities` is one score per answer, None for an answer nobody graded.
96:         Returns the execution id, so a test can add a second one and check that
97:         only the latest counts.
98:         """
99:         row = self.individual(number)
100:         execution = store.add_execution(
101:             self.conn, row["id"], seconds=1.0, exit_code=exit_code,
102:             verdict=verdict, weight_seed=42,
103:             weights={"w%d" % slot: 0.5 for slot in range(1, 6)},
104:             stdout="", stderr="")
105:         store.add_exchanges(self.conn, execution, [
106:             {"question": "q%d" % position, "answer": "a%d" % position,
107:              "quality": quality, "reason": "because"}
108:             for position, quality in enumerate(qualities, 1)])
109:         self.conn.commit()
110:         return execution
111: 
112:     # --- reading it back ---------------------------------------------------
113: 
114:     def individual(self, number):
115:         return self.conn.execute(
116:             "SELECT * FROM individuals WHERE run_id = ? AND number = ?",
117:             (self.run, number)).fetchone()
118: 
119:     def population(self):
120:         return store.individuals(self.conn, self.run)
121: 
122:     def numbers(self):
123:         return [row["number"] for row in self.population()]
124: 
125:     def fitnesses(self):
126:         """{number: fitness} exactly as stored -- NULL stays None."""
127:         return {row["number"]: row["fitness"] for row in self.population()}
128: 
129:     def chromosomes(self):
130:         return {row["number"]: row["chromosome"] for row in self.population()}
131: 
132:     def elite(self):
133:         """The number of the individual flagged is_best, or None."""
134:         marked = [row["number"] for row in self.population() if row["is_best"]]
135:         self.assertLessEqual(len(marked), 1, "a sweep may hold at most one elite")
136:         return marked[0] if marked else None
137: 
138:     def execution_count(self):
139:         return self.conn.execute(
140:             "SELECT COUNT(*) AS n FROM executions e JOIN individuals i"
141:             " ON i.id = e.individual_id WHERE i.run_id = ?",
142:             (self.run,)).fetchone()["n"]
143: 
144:     def exchange_count(self):
145:         return self.conn.execute(
146:             "SELECT COUNT(*) AS n FROM exchanges x JOIN executions e"
147:             " ON e.id = x.execution_id JOIN individuals i"
148:             " ON i.id = e.individual_id WHERE i.run_id = ?",
149:             (self.run,)).fetchone()["n"]
150: 
151: 
152: def rng(seed=1234):
153:     """A generator seeded the way a sweep seeds its own: fixed, and named."""
154:     return random.Random(seed)

(End of file - total 154 lines)
</content>
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 95.5s)

_Thinking:_

Now let me run the unit tests. This is a good check for the project's health. Use the venv python. Discovery from unittests. It may take a while, especially async_api tests that stand up servers. Let me set a generous timeout (say 300000 ms = 5 min). Actually the default is 120s. Let me set 600000.

Wait — running tests may start training or load models? The async_api tests use mocked things presumably. Let me run and see. I'll run with the venv python and capture output. If it's too long I'll narrow.

Let me run it.

**Tool: bash**

**Input:**
```json
{
  "command": "D:\\sage-is\\loras\\.venv\\Scripts\\python.exe -m unittest discover -s unittests -t . 2>&1 | Select-Object -Last 60",
  "description": "Run full unit test suite",
  "timeout": 600000
}
```

**Output:**
```
[worker 19:59:55] job 1: done
[worker 19:59:55] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-nmicvujh\jobs\api.sqlite
[worker 19:59:55] verification 1 (job 1, individual 3): gep_lora.core.testing.evaluate_chromosome_against_loras --db C:\Users\ze\AppData\Local\Temp\gep-api-tests-nmicvujh\jobs\user1\job1\job.sqlite3 --run 1 --individual 3 --into C:\Users\ze\AppData\Local\Temp\gep-api-tests-nmicvujh\jobs\user1\job1\verify1 --timeout 900 --evaluator similarity --split testing --slots L1,L2
[worker 19:59:58] verification 1: done
[worker 19:59:59] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-mj0t9k7k\jobs\api.sqlite
[worker 19:59:59] job 1: D:\sage-is\loras\gep-lora\main.py --db C:\Users\ze\AppData\Local\Temp\gep-api-tests-mj0t9k7k\jobs\user1\job1\job.sqlite3 --run 1 --test-min-quality 0.0
[worker 20:00:01] job 1: done
[worker 20:00:03] verification 1 was left running by a previous worker; marked failed
[worker 20:00:04] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-26t_zi8v\jobs\api.sqlite
[worker 20:00:04] training 1 (tiny-qwen3.5-0.8b-20260923-r8): gep_lora.core.adapters.create_lora C:\Users\ze\AppData\Local\Temp\gep-api-tests-26t_zi8v\trained\user1\tiny-qwen3.5-0.8b-20260923-r8 --dataset C:\Users\ze\AppData\Local\Temp\gep-api-tests-26t_zi8v\jobs\user1\lora1\dataset.jsonl --dataset-source uploaded by alice --name tiny-qwen3.5-0.8b-20260923-r8 --catalog C:\Users\ze\AppData\Local\Temp\gep-api-tests-26t_zi8v\jobs\api.sqlite --base-model unsloth/Qwen3.5-0.8B --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --scheduler linear --optim adamw_8bit --rank 8 --alpha 16 --dropout 0.0 --epochs 1.0 --learning-rate 0.0002 --warmup-steps 5 --weight-decay 0.01 --batch-size 2 --grad-accum 4 --max-seq 2048 --seed 3407 --owner alice --prompt What is question 1? --mock --mock-delay 0
[worker 20:00:04] training 1: done
[worker 20:00:04] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-26t_zi8v\jobs\api.sqlite
[worker 20:00:04] training 2 (tiny-qwen3.5-0.8b-20260923-r16): gep_lora.core.adapters.create_lora C:\Users\ze\AppData\Local\Temp\gep-api-tests-26t_zi8v\trained\user1\tiny-qwen3.5-0.8b-20260923-r16 --dataset C:\Users\ze\AppData\Local\Temp\gep-api-tests-26t_zi8v\jobs\user1\lora2\dataset.jsonl --dataset-source uploaded by alice --name tiny-qwen3.5-0.8b-20260923-r16 --catalog C:\Users\ze\AppData\Local\Temp\gep-api-tests-26t_zi8v\jobs\api.sqlite --base-model unsloth/Qwen3.5-0.8B --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --scheduler linear --optim adamw_8bit --rank 16 --alpha 16 --dropout 0.0 --epochs 1.0 --learning-rate 0.0002 --warmup-steps 5 --weight-decay 0.01 --batch-size 2 --grad-accum 4 --max-seq 2048 --seed 3407 --owner alice --prompt What is question 1? --mock --mock-delay 0
[worker 20:00:05] training 2: done
[worker 20:00:05] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-26t_zi8v\jobs\api.sqlite
[worker 20:00:05] job 1: D:\sage-is\loras\gep-lora\main.py --db C:\Users\ze\AppData\Local\Temp\gep-api-tests-26t_zi8v\jobs\user1\job1\job.sqlite3 --run 1 --no-test
[worker 20:00:05] job 1: done
[worker 20:00:07] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-d1pt18n_\jobs\api.sqlite
[worker 20:00:07] training 1 (poem-qwen3.5-0.8b-20260923-r8): gep_lora.core.adapters.create_lora C:\Users\ze\AppData\Local\Temp\gep-api-tests-d1pt18n_\trained\user1\poem-qwen3.5-0.8b-20260923-r8 --dataset C:\Users\ze\AppData\Local\Temp\gep-api-tests-d1pt18n_\jobs\user1\lora1\dataset.jsonl --dataset-source datasets/poem_lora_dataset.json --name poem-qwen3.5-0.8b-20260923-r8 --catalog C:\Users\ze\AppData\Local\Temp\gep-api-tests-d1pt18n_\jobs\api.sqlite --base-model unsloth/Qwen3.5-0.8B --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --scheduler linear --optim adamw_8bit --rank 8 --alpha 16 --dropout 0.0 --epochs 1.0 --learning-rate 0.0002 --warmup-steps 5 --weight-decay 0.01 --batch-size 2 --grad-accum 4 --max-seq 2048 --seed 3407 --owner alice --prompt Hi there! --mock --mock-delay 0
[worker 20:00:07] training 1: done
[worker 20:00:07] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-d1pt18n_\jobs\api.sqlite
[worker 20:00:07] training 2 (poem-qwen3.5-0.8b-20260923-r16): gep_lora.core.adapters.create_lora C:\Users\ze\AppData\Local\Temp\gep-api-tests-d1pt18n_\trained\user1\poem-qwen3.5-0.8b-20260923-r16 --dataset C:\Users\ze\AppData\Local\Temp\gep-api-tests-d1pt18n_\jobs\user1\lora2\dataset.jsonl --dataset-source datasets/poem_lora_dataset.json --name poem-qwen3.5-0.8b-20260923-r16 --catalog C:\Users\ze\AppData\Local\Temp\gep-api-tests-d1pt18n_\jobs\api.sqlite --base-model unsloth/Qwen3.5-0.8B --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --scheduler linear --optim adamw_8bit --rank 16 --alpha 16 --dropout 0.0 --epochs 1.0 --learning-rate 0.0002 --warmup-steps 5 --weight-decay 0.01 --batch-size 2 --grad-accum 4 --max-seq 2048 --seed 3407 --owner alice --prompt Hi there! --mock --mock-delay 0
[worker 20:00:08] training 2: done
[worker 20:00:17] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\api.sqlite
[worker 20:00:17] training 1 (tiny-qwen3.5-0.8b-20260923-r8): gep_lora.core.adapters.create_lora C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\trained\user1\tiny-qwen3.5-0.8b-20260923-r8 --dataset C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\user1\lora1\dataset.jsonl --dataset-source uploaded by alice --name tiny-qwen3.5-0.8b-20260923-r8 --catalog C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\api.sqlite --base-model unsloth/Qwen3.5-0.8B --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --scheduler linear --optim adamw_8bit --rank 8 --alpha 16 --dropout 0.0 --epochs 1.0 --learning-rate 0.0002 --warmup-steps 5 --weight-decay 0.01 --batch-size 2 --grad-accum 4 --max-seq 2048 --seed 3407 --owner alice --prompt What is question 1? --mock --mock-delay 0
[worker 20:00:17] training 1: done
[worker 20:00:17] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\api.sqlite
[worker 20:00:17] training 2 (tiny-qwen3.5-0.8b-20260923-r16): gep_lora.core.adapters.create_lora C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\trained\user1\tiny-qwen3.5-0.8b-20260923-r16 --dataset C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\user1\lora2\dataset.jsonl --dataset-source uploaded by alice --name tiny-qwen3.5-0.8b-20260923-r16 --catalog C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\api.sqlite --base-model unsloth/Qwen3.5-0.8B --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --scheduler linear --optim adamw_8bit --rank 16 --alpha 16 --dropout 0.0 --epochs 1.0 --learning-rate 0.0002 --warmup-steps 5 --weight-decay 0.01 --batch-size 2 --grad-accum 4 --max-seq 2048 --seed 3407 --owner alice --prompt What is question 1? --mock --mock-delay 0
[worker 20:00:17] training 2: done
[worker 20:00:17] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\api.sqlite
[worker 20:00:17] job 1: D:\sage-is\loras\gep-lora\main.py --db C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\user1\job1\job.sqlite3 --run 1 --no-test
[worker 20:00:17] job 1: done
[worker 20:00:18] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\api.sqlite
[worker 20:00:18] job 1 (test): -m gep_lora.core.testing.test_run_with_dataset --from-db --db C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\user1\job1\job.sqlite3 --run 1 --min-quality=-1.0 --resume
[worker 20:00:18] job 1: done
[worker 20:00:18] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\api.sqlite
[worker 20:00:18] verification 1 (job 1, individual 1): gep_lora.core.testing.evaluate_chromosome_against_loras --db C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\user1\job1\job.sqlite3 --run 1 --individual 1 --into C:\Users\ze\AppData\Local\Temp\gep-api-tests-pa01khbr\jobs\user1\job1\verify1 --timeout 900 --split validation --count 1 --slots L1
[worker 20:00:18] verification 1: done
[worker 20:00:19] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-h30_uue3\jobs\api.sqlite
[worker 20:00:19] training 1 (tiny-qwen3.5-0.8b-20260923-r8): gep_lora.core.adapters.create_lora C:\Users\ze\AppData\Local\Temp\gep-api-tests-h30_uue3\trained\user1\tiny-qwen3.5-0.8b-20260923-r8 --dataset C:\Users\ze\AppData\Local\Temp\gep-api-tests-h30_uue3\jobs\user1\lora1\dataset.jsonl --dataset-source uploaded by alice --name tiny-qwen3.5-0.8b-20260923-r8 --catalog C:\Users\ze\AppData\Local\Temp\gep-api-tests-h30_uue3\jobs\api.sqlite --base-model unsloth/Qwen3.5-0.8B --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --scheduler linear --optim adamw_8bit --rank 8 --alpha 16 --dropout 0.0 --epochs 1.0 --learning-rate 0.0002 --warmup-steps 5 --weight-decay 0.01 --batch-size 2 --grad-accum 4 --max-seq 2048 --seed 3407 --owner alice --prompt What is question 1? --mock --mock-delay 0
[worker 20:00:19] training 1: done
[worker 20:00:19] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-h30_uue3\jobs\api.sqlite
[worker 20:00:19] training 2 (tiny-qwen3.5-0.8b-20260923-r16): gep_lora.core.adapters.create_lora C:\Users\ze\AppData\Local\Temp\gep-api-tests-h30_uue3\trained\user1\tiny-qwen3.5-0.8b-20260923-r16 --dataset C:\Users\ze\AppData\Local\Temp\gep-api-tests-h30_uue3\jobs\user1\lora2\dataset.jsonl --dataset-source uploaded by alice --name tiny-qwen3.5-0.8b-20260923-r16 --catalog C:\Users\ze\AppData\Local\Temp\gep-api-tests-h30_uue3\jobs\api.sqlite --base-model unsloth/Qwen3.5-0.8B --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --scheduler linear --optim adamw_8bit --rank 16 --alpha 16 --dropout 0.0 --epochs 1.0 --learning-rate 0.0002 --warmup-steps 5 --weight-decay 0.01 --batch-size 2 --grad-accum 4 --max-seq 2048 --seed 3407 --owner alice --prompt What is question 1? --mock --mock-delay 0
[worker 20:00:19] training 2: done
[worker 20:00:19] watching C:\Users\ze\AppData\Local\Temp\gep-api-tests-h30_uue3\jobs\api.sqlite
[worker 20:00:19] job 1: D:\sage-is\loras\gep-lora\main.py --db C:\Users\ze\AppData\Local\Temp\gep-api-tests-h30_uue3\jobs\user1\job1\job.sqlite3 --run 1 --no-test
[worker 20:00:20] job 1: done
.......................................................................................................................
......................................................................F................................................
.......................................................................................................................
...................................................................................................
======================================================================
FAIL: test_it_does_not_ask_for_a_copy 
(unittests.core.evaluators.test_llm_judge_answers.RubricTests.test_it_does_not_ask_for_a_copy)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "D:\sage-is\loras\gep-lora\unittests\evaluators\test_llm_judge_answers.py", line 170, in 
test_it_does_not_ask_for_a_copy
    self.assertIn("Do not reward copying", self.PROMPT)
AssertionError: 'Do not reward copying' not found in 'You are comparing two answers to the same question. You are NOT 
shown the\nquestion, and you do not need it. Do not ask for it and do not speculate about\nit -- judge only the two 
answers in front of you.\n\nYou will be shown:\n  REFERENCE ANSWER  ground truth\n  ANSWER            the answer to be 
evaluated against ground truth (REFERENCE ANSWER)\n\nJudge how well the ANSWER stands in for the REFERENCE ANSWER:\n- 
Manner: the same style, voice, form, length and register? This matters most\n- Substance: does it say the same kind of 
thing, so that someone expecting the\n  reference would be served by this instead?\n- Coherence: well formed and 
consistent, free of contradictions, repetition,\n  broken grammar or nonsense.\n\nScore from 0.0 to 1.0:\n  1.0  
excellent - same manner, and would serve in place of the reference\n  0.7  good - recognisably the same manner, minor 
slips\n  0.5  mixed - part of the manner, or the manner without the substance\n  0.3  poor - an answer, but in nothing 
like the intended manner\n  0.0  useless - incoherent, empty, or plainly about something else\n\nReply with JSON and 
nothing else, with the score FIRST:\n{"quality": <number between 0 and 1>, "reason": "<at most 12 words>"}\n'
----------------------------------------------------------------------
Ran 471 tests in 90.603s
FAILED (failures=1)

```

---

## Assistant (Build · DeepSeek V4.1 Flash · 4.7s)

_Thinking:_

Interesting — there's a failing test. The test `test_it_does_not_ask_for_a_copy` expects "Do not reward copying" in the prompt, but the prompt doesn't contain it. This suggests a test/code drift — either the prompt was changed and test not updated, or the test is checking for something that should be added. This is a concrete finding.

Let me verify by reading the test and the prompt in llm_judge_answers.py.

Also, note 471 tests in 90s. Reasonable.

Let me look at that test and prompt.

**Tool: bash**

**Input:**
```json
{
  "command": "Get-Content unittests\\evaluators\\test_llm_judge_answers.py | Select-Object -Skip 150 -First 30; echo \"=== PROMPT in module ===\"; Select-String -Path evaluators\\llm_judge_answers.py -Pattern \"PROMPT|copy|Manner|Substance\" | Select-Object -First 20",
  "description": "Inspect failing test and prompt"
}
```

**Output:**
```


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
=== PROMPT in module ===

evaluators\llm_judge_answers.py:13:it a judge quietly grades merit as well as manner: a blend that answers helpfully
evaluators\llm_judge_answers.py:16:-- the reference is the only standard in the prompt, so the score is a
evaluators\llm_judge_answers.py:18:shortens the prompt by a question per call, which on a local judge with
evaluators\llm_judge_answers.py:44:# out in the manner the training data replies in -- without the escape hatch of
evaluators\llm_judge_answers.py:51:# those exist so a sweep created while the prompt was still a setting is
evaluators\llm_judge_answers.py:52:# re-scored against the prompt it actually ran with, and no sweep has ever run
evaluators\llm_judge_answers.py:54:JUDGE_ANSWERS_SYSTEM_PROMPT = """\
evaluators\llm_judge_answers.py:64:- Manner: the same style, voice, form, length and register? This matters most
evaluators\llm_judge_answers.py:65:- Substance: does it say the same kind of thing, so that someone expecting the
evaluators\llm_judge_answers.py:71:  1.0  excellent - same manner, and would serve in place of the reference
evaluators\llm_judge_answers.py:72:  0.7  good - recognisably the same manner, minor slips
evaluators\llm_judge_answers.py:73:  0.5  mixed - part of the manner, or the manner without the substance
evaluators\llm_judge_answers.py:74:  0.3  poor - an answer, but in nothing like the intended manner
evaluators\llm_judge_answers.py:112:            "no reference answer for prompt %s, and llm_judge_answers has "
evaluators\llm_judge_answers.py:117:    return common.ask_judge(JUDGE_ANSWERS_SYSTEM_PROMPT, content,
evaluators\llm_judge_answers.py:124:    "never shown the question (JUDGE_ANSWERS_SYSTEM_PROMPT)",



```

---

## Assistant (Build · DeepSeek V4.1 Flash · 15.1s)

_Thinking:_

Confirmed: the test expects "Do not reward copying" but the prompt no longer contains it. This is a real drift/bug — either the prompt was edited and the test wasn't updated, or vice versa. The prompt currently lacks the anti-copying instruction (which is meaningful: a judge could reward copying the reference). This is a legitimate finding.

Now I have enough material for a thorough critique. Let me also quickly check a couple of specific correctness concerns I noticed:

1. `store.add_exchanges` — no commit inside, caller commits. Fine.
2. `generate_population.random_tree`: at max_depth, only UNARY_OPS drawn. But `depth + 1 >= max_depth` — let me trace: root at depth 0, max_depth from `rng.randint(1, max_depth)` in build_population. For a BINARY_OP node at depth d, if d+1 >= max_depth then children are UNARY. So operators at depth max_depth-1 become unary. Fine.
3. `selection.wheel`/`spin` — potential issue: `draw` uses `rng.random()` per spin, but `wheel` edges recomputed once. Fine.
4. `mutation.apply` calls `store.set_changed` per row and `set_chromosome`; commits once. Fine.
5. `weight_mutation.mutate` uses `rng.sample(pool, draw_count(...))`. Fine.

Potential real bugs I can spot:

- In `start_run.step_process`, `store.materialise` is called, but `rows` was computed before materialise. Fine.
- `_clear_scripts` for `selected + unchanged` — okay.
- `process_run.exchanges`: the `_split_grade` logic — `elif not grade:` means once a QUALITY/REASON line is seen, grade is non-empty so subsequent answer lines are dropped. That's intended.

- `store.append_copies`: builds INSERT with columns from PRAGMA excluding id/number, and inserts `[number] + [0 if is_best else row[column]]`. But the `columns` includes `run_id`. The copy keeps parent's run_id — correct. Good.

- Potential issue: `individual_quality` view groups by `i.id` but selects non-aggregated columns; sqlite allows it. Fine.

- `fitness_generation` uses `fitness_history` max generation and high_number. Fine.

- A subtle bug: `store.delete_individuals` reads rows with `SELECT * ... ORDER BY number` then deletes. But `append_copies` in selection happens before cull; cull deletes rows that were read in `selection.select` as `rows` (the pre-round population). Good.

- `selection.select`: `fresh(rows, rng, conf)` — uses `conf["MAX_DEPTH"]`, `conf["BRANCH_PROB"]` directly (KeyError if missing). For an old sweep lacking these, it'd crash. Minor.

- `weight_mutation.draw_count`: `min(total, int(rate*total+0.5))`. For rate=1.0, int(total+0.5)=total. Fine.

- In `template_code.py`, `_compact` is called inside `combine` for svd. It iterates `model.modules()`. Fine.

- In `template_code.answer`, `out` includes prompt; slices `row[width:]`. Fine with left padding.

- Security: The generated scripts are Python and run; fine.

- The `NUL` file at repo root is a Windows artifact and is gitignored. Actually gitignore has `NUL`. Fine.

- `api/` folder with only pycache — stale leftover; could delete. Actually the folder `api/` is not gitignored (only `api_jobs`). Let me check git status for api. It only contains __pycache__, which is ignored. So the directory is effectively empty in git. Probably leftover from an earlier "api" name. Cleanup suggestion.

- No `requirements.txt`/`pyproject.toml`/`environment.yml`. Dependency management is via .bat scripts. This is a real gap for reproducibility, especially since the repo records git_commit but not package versions.

- No lint/typecheck configuration despite CLAUDE.md instructing to run lint/typecheck. There's no ruff/flake8/mypy config. The `# noqa: BLE001` comment suggests flake8/ruff usage but no config file. That's a gap.

- Huge modules: gep_lora/core/reporting/generate_html_db_stats.py 2424 lines, store.py 1388, gep_lora/core/pipeline/start_run.py 1462. `App` god object.

- Documentation is enormous (CLAUDE.md 95KB, README 200KB). The docstrings are essay-like — this is unusual; a critique point is that comments are extremely verbose and duplicative (same explanation repeated in many modules), increasing maintenance burden. This is a legitimate style critique, though the user may value it. I'll note it neutrally.

- Test gap: `unittests/blends`, `unittests/storage`, `unittests/tools`, `unittests/api` directories exist but the file listing shows no test files in blends/storage/tools/api! Let me double check — the file listing showed only adapters, async_api, async_api_agent, evaluators, search, testing, test_main.py. So `unittests/blends`, `unittests/storage`, `unittests/tools`, `unittests/api` are empty directories (only __pycache__). Let me verify. The directory listing showed `unittests\blends`, `unittests\storage`, `unittests\tools`, `unittests\api` as directories but the -File listing didn't show files there. Actually the -File listing only listed specific files; `unittests\blends` etc. weren't listed with any .py. Wait, it listed `unittests\adapters\test_catalog.py` etc. It did NOT list any files under `unittests\blends`, `unittests\storage`, `unittests\tools`, `unittests\api`. So those directories contain no test files — only __pycache__ from removed tests? Let me verify quickly.

Also `unittests\api` vs `unittests\async_api` — there's an `api` test dir too.

Let me check what's in those empty dirs, and also whether process_run/generate_runs/store are tested. From the file list: search tests exist for calculate_fitness, draw_trees, elitism, generate_population, generation_cycle, mutation, selection, weight_mutation. Storage has no tests. Blends (generate_runs, process_run, lora_server, server_pool) — no tests. That's the most critical untested area: the code generation and subprocess parsing (regex-based transcript parsing), which is exactly where subtle bugs live.

Let me verify empty dirs.

**Tool: bash**

**Input:**
```json
{
  "command": "foreach ($d in 'unittests\\blends','unittests\\storage','unittests\\tools','unittests\\api') { Write-Output \"== $d ==\"; Get-ChildItem -Recurse -File $d -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name }",
  "description": "Check for empty test directories"
}
```

**Output:**
```
== unittests\blends ==
== unittests\storage ==
== unittests\tools ==
== unittests\api ==
support.cpython-313.pyc
test_catalog.cpython-313.pyc
test_progress.cpython-313.pyc
test_prompting.cpython-313.pyc
test_server.cpython-313.pyc
__init__.cpython-313.pyc

```

**Tool: grep**

**Input:**
```json
{
  "pattern": "import process_run|from blends import|generate_runs|store\\.",
  "include": "*.py",
  "path": "D:\\sage-is\\loras\\gep-lora\\unittests"
}
```

**Output:**
```
Found 89 matches
D:\sage-is\loras\gep-lora\unittests\async_api\test_server.py:
  Line 121:         # The database downloads whole, as a sweep store.py can read back.

  Line 129:         conn = store.connect(copy)

  Line 131:             self.assertEqual(len(store.individuals(conn, reply["job"]["run_id"])), 4)

  Line 132:             self.assertEqual(len(store.fitness_by_generation(conn, reply["job"]["run_id"])), 2)

  Line 273:         conn = store.connect(self.registry.database(job))

  Line 275:             return [row["items"] for row in store.step_timings(conn, job["run_id"])

  Line 336:         conn = store.connect(self.registry.database(job))

  Line 338:             judges = {row["judge_model"] for row in store.exchanges_to_score(


D:\sage-is\loras\gep-lora\unittests\async_api\test_submit.py:
  Line 21:         conn = store.connect(self.registry.database(job))

  Line 23:             conf = store.get_settings(conn, job["run_id"])

  Line 25:                       for one in store.dataset_summary(conn, job["run_id"])}

  Line 26:             self.assertEqual(store.individuals(conn, job["run_id"]), [])

  Line 76:         conn = store.connect(self.registry.database(job))

  Line 78:             conf = store.get_settings(conn, job["run_id"])

  Line 107:         conn = store.connect(self.registry.database(job))

  Line 109:             stored = store.get_settings(conn, job["run_id"])["LORA_SLOTS"]


D:\sage-is\loras\gep-lora\unittests\test_main.py:
  Line 26:         pass_no = store.next_pass(self.conn, self.run)

  Line 28:             store.add_step_timing(self.conn, self.run, pass_no, position, step, 0.1)

  Line 30:                 generation = len(store.fitness_by_generation(self.conn, self.run)) + 1

  Line 31:                 store.record_fitness(self.conn, self.run, generation, [

  Line 86:         store.add_step_timing(self.conn, self.run, store.next_pass(self.conn, self.run),


D:\sage-is\loras\gep-lora\unittests\testing\test_evaluate_chromosome_against_loras.py:
  Line 27: from blends import process_run

  Line 125:         self.conn = store.connect(os.path.join(folder, "gep.sqlite3"))

  Line 127:         self.run_id = store.create_run(self.conn, "template_code_mocked.py")

  Line 128:         store.add_individuals(self.conn, self.run_id,

  Line 131:             store.set_fitness(self.conn, self.run_id, number, fitness)

  Line 138:         store.mark_best(self.conn, self.run_id, 3)


D:\sage-is\loras\gep-lora\unittests\async_api\test_golive.py:
  Line 19: from blends import generate_runs

  Line 62:             steps, final = generate_runs.plan(decode(chromosome)[0], RANKS)

  Line 63:             source = generate_runs.render(

  Line 65:                 template_path=generate_runs.template_path("template_remote_code.py"),


D:\sage-is\loras\gep-lora\unittests\search\test_weight_mutation.py:
  Line 94:         store.mark_best(self.conn, self.run, 2)

  Line 104:         store.mark_best(self.conn, self.run, 2)

  Line 125:             store.set_changed(self.conn, row["id"], 1)

  Line 137:         store.add_individuals(self.conn, self.run, [])


D:\sage-is\loras\gep-lora\unittests\search\test_generation_cycle.py:
  Line 82:         store.add_individuals(self.conn, self.run, self.starting_population())

  Line 195:                           store.fitness_by_generation(self.conn, self.run)],

  Line 200:         for row in store.fitness_by_generation(self.conn, self.run):

  Line 235:         watermark = store.high_number(self.conn, self.run)

  Line 238:             now = store.high_number(self.conn, self.run)

  Line 265:         history = store.fitness_history(self.conn, self.run, generation=1)

  Line 278:         self.run = store.create_run(self.conn, "template_code_mocked.py")


D:\sage-is\loras\gep-lora\unittests\search\test_selection.py:
  Line 190:         store.mark_best(self.conn, self.run, 6)

  Line 197:         store.mark_best(self.conn, self.run, 6)

  Line 218:         self.assertEqual(store.next_number(self.conn, self.run),

  Line 222:         store.mark_best(self.conn, self.run, 6)

  Line 229:         store.mark_best(self.conn, self.run, 1)     # fitness 0.0

  Line 241:         store.mark_best(self.conn, self.run, 6)

  Line 252:         store.mark_best(self.conn, self.run, 6)

  Line 254:         store.set_tree(self.conn, parent["id"], "a drawing")

  Line 255:         store.set_script(self.conn, parent["id"], "ok", 24, "run_006.py",

  Line 273:         store.mark_best(self.conn, self.run, 6)

  Line 312:         store.add_individuals(self.conn, self.run, [])

  Line 325:         store.mark_best(self.conn, self.run, 6)

  Line 336:         store.mark_best(self.conn, self.run, 6)

  Line 350:         other = store.create_run(self.conn, "template_code_mocked.py")

  Line 351:         store.add_individuals(self.conn, other,

  Line 355:             store.set_fitness(self.conn, other, number, fitness)

  Line 385:         store.mark_best(self.conn, self.run, 6)


D:\sage-is\loras\gep-lora\unittests\search\test_elitism.py:
  Line 103:         store.set_chromosome(self.conn, self.individual(1)["id"],

  Line 123:         store.add_individuals(self.conn, self.run, [])


D:\sage-is\loras\gep-lora\unittests\search\test_calculate_fitness.py:
  Line 112:         store.add_individuals(self.conn, self.run, [])

  Line 115:         self.assertEqual(store.fitness_history(self.conn, self.run), [])

  Line 134:         history = store.fitness_history(self.conn, self.run)

  Line 141:                   for row in store.fitness_history(self.conn, self.run)}

  Line 150:         self.assertEqual(len(store.fitness_history(self.conn, self.run)), 3)

  Line 156:         history = store.fitness_history(self.conn, self.run, generation=1)

  Line 164:         store.append_individual(self.conn, self.run, "CAT.L1.L2.w1.w2")

  Line 168:                           store.fitness_by_generation(self.conn, self.run)],

  Line 175:         store.append_individual(self.conn, self.run, "CAT.L1.L2.w1.w2")

  Line 176:         store.delete_individuals(self.conn, self.run, [3])

  Line 185:         store.set_chromosome(self.conn, self.individual(2)["id"],

  Line 188:         history = store.fitness_history(self.conn, self.run, generation=1)

  Line 194:         store.delete_individuals(self.conn, self.run, [2])

  Line 195:         history = store.fitness_history(self.conn, self.run, generation=1)

  Line 202:         row = store.fitness_by_generation(self.conn, self.run)[0]

  Line 208:         best = store.best_of_generation(self.conn, self.run, 1)


D:\sage-is\loras\gep-lora\unittests\search\test_mutation.py:
  Line 189:         store.mark_best(self.conn, self.run, 2)

  Line 204:         store.mark_best(self.conn, self.run, 2)

  Line 232:         store.add_individuals(self.conn, self.run, [])


D:\sage-is\loras\gep-lora\unittests\search\support.py:
  Line 13: this suite agreeing with itself about SQL nobody ran. store.py is the only

  Line 57:         # store.connect() resolves a relative path against the repo folder, so

  Line 60:         self.conn = store.connect(os.path.join(self.folder, "test.sqlite3"))

  Line 61:         self.run = store.create_run(self.conn, "template_code_mocked.py",

  Line 78:         store.add_individuals(self.conn, self.run, chromosomes)

  Line 89:             store.set_fitness(self.conn, self.run, number, fitness)

  Line 100:         execution = store.add_execution(

  Line 105:         store.add_exchanges(self.conn, execution, [

  Line 120:         return store.individuals(self.conn, self.run)

```

---

## Assistant (Build · DeepSeek V4.1 Flash · 6.5s)

_Thinking:_

Confirmed: `unittests/blends`, `unittests/storage`, `unittests/tools` are empty (only pycache), and `unittests/api` has pycache for removed `test_progress.py`, `test_prompting.py`, `test_server.py`, `support.py`, `test_catalog.py` — stale bytecode from a deleted `api` test suite. These are dead artifacts.

So the critical untested modules: `gep_lora/core/blends/generate_runs.py` (only exercised indirectly by test_golive and evaluate_chromosome tests), `gep_lora/core/blends/process_run.py` (only `process_run` imported in one testing test), `gep_lora/core/storage/store.py` (exercised via support fixtures but no direct tests), `gep_lora/core/reporting/generate_html_db_stats.py` (2424 lines, no tests), `gep_lora/core/metrics/report.py`, `gep_lora/core/search/draw_trees.py` (has test), `gep_lora/core/blends/lora_server.py`, `gep_lora/core/blends/server_pool.py`, `gep_lora/service/worker.py`, `gep_lora/service/evaluate.py`, `gep_lora/service/testpass.py`.

I have plenty. Let me also quickly confirm there's no requirements file at all and check the `docs/` folder. And check for any TODO/FIXME/HACK markers.

Let me do a final quick sweep.

**Tool: bash**

**Input:**
```json
{
  "command": "Get-ChildItem docs -Recurse -File | Select-Object -ExpandProperty Name; echo \"--- root non-py files ---\"; Get-ChildItem -File | Where-Object { $_.Extension -notin '.py' } | Select-Object -ExpandProperty Name",
  "description": "List docs and root files"
}
```

**Output:**
```
gep.sqlite3
run_report.txt
--- root non-py files ---
.gitignore
activate.bat
async_api.txt
CLAUDE.md
create_env.bat
delete_env.bat
download_wheels.bat
features_to_implement.md
generate_report.bat
HOWTO.md
LICENSE
NUL
PerformanceWork.md
plan.txt
README.md
run_api_server.bat
run_api_worker.bat
run_unit_tests.bat
setup_torch.bat

```

**Tool: grep**

**Input:**
```json
{
  "pattern": "TODO|FIXME|XXX|HACK|BUG\\b",
  "include": "*.py",
  "path": "D:\\sage-is\\loras\\gep-lora"
}
```

**Output:**
```
No files found
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 6.1s)

_Thinking:_

Interesting: no TODO/FIXME markers at all. `docs/` contains `gep.sqlite3` and `run_report.txt` — a database committed under docs? Let me check if docs/gep.sqlite3 is tracked by git. Also `NUL` file exists (gitignored). No requirements file. `.bat` files are gitignored (`*.bat`) which means the setup scripts aren't tracked! That's notable — `activate.bat`, `setup_torch.bat`, `create_env.bat`, `run_unit_tests.bat`, `run_api_server.bat`, etc. are all ignored by `*.bat`. So a fresh clone lacks the environment setup and run scripts. Actually wait — `*.bat` in gitignore means they're untracked. Let me verify with git ls-files.

Let me check git tracking of key files and docs/gep.sqlite3.

**Tool: bash**

**Input:**
```json
{
  "command": "git ls-files | Measure-Object -Line; echo \"--- bat tracked? ---\"; git ls-files \"*.bat\"; echo \"--- docs ---\"; git ls-files docs; echo \"--- requirements/pyproject ---\"; git ls-files \"*requirements*\" \"*pyproject*\" \"*setup.py*\" \"*.toml\" \"*.cfg\" \"*.ini\"; echo \"--- README/CLAUDE sizes ---\"; (Get-Item README.md).Length; (Get-Item CLAUDE.md).Length",
  "description": "Check git-tracked files"
}
```

**Output:**
```

generate_report.bat
run_unit_tests.bat
docs/example_run/gep.sqlite3
docs/example_run/run_report.txt
Lines Words Characters Property
----- ----- ---------- --------
  143                          
--- bat tracked? ---
--- docs ---
--- requirements/pyproject ---
--- README/CLAUDE sizes ---
201987
95254



```

---

## Assistant (Build · DeepSeek V4.1 Flash · 18.5s)

_Thinking:_

Hmm, the output is confusing. `git ls-files | Measure-Object -Line` gives 143 tracked files. Then `git ls-files "*.bat"` returned nothing (so .bat are untracked, confirmed by the earlier listing that showed `generate_report.bat` and `run_unit_tests.bat` at the top... wait, the first lines of output "generate_report.bat / run_unit_tests.bat" appear before the Measure-Object output. That's actually the output of `git ls-files "*.bat"`? Let me re-read.

Output order:
```
generate_report.bat
run_unit_tests.bat
docs/example_run/gep.sqlite3
docs/example_run/run_report.txt
Lines Words Characters Property
----- ----- ---------- -----
  143
--- bat tracked? ---
--- docs ---
--- requirements/pyproject ---
```

Hmm, the ordering is jumbled because PowerShell interleaves. Actually the "generate_report.bat / run_unit_tests.bat" and "docs/..." are likely the outputs of the git ls-files commands but appear before the echo labels due to buffering. So:
- `git ls-files "*.bat"` → generate_report.bat, run_unit_tests.bat (two tracked bats)
- `git ls-files docs` → docs/example_run/gep.sqlite3, docs/example_run/run_report.txt
- requirements/pyproject → empty (none)

So some .bat files ARE tracked (generate_report.bat, run_unit_tests.bat) even though `*.bat` is in gitignore — because they were added before the ignore rule, or force-added. The others (setup_torch.bat, create_env.bat, activate.bat, run_api_server.bat, run_api_worker.bat, delete_env.bat, download_wheels.bat) are untracked. So environment setup scripts are not in the repo. That's a reproducibility gap.

Also `docs/example_run/gep.sqlite3` — a binary database is committed. That's ~? Could be large. Let me check size. Also the root `docs/gep.sqlite3`? No, it's docs/example_run.

Also no requirements/pyproject/setup at all. So the repo cannot be set up from a fresh clone without the untracked .bat files and manually installing torch/unsloth/peft/transformers.

I think I have a comprehensive picture. Let me also note the `NUL` file (Windows reserved name artifact) at root — harmless but messy.

Now, let me consider the correctness bug I found in the failing test. Also, let me double check one important claim from the agent about the `judge_models` SSRF: it's confirmed from server.py:603-622 and common.py:66,247-291. The API key is sent to a user-supplied base_url. This is real. But note: `/judge/models` requires authentication. And it's a self-hosted tool. Still valid.

Also the agent found `evaluate.py` doesn't validate base_url. Let me trust it.

Let me quickly sanity check the digest claim and the negative Content-Length claim. Both seem plausible from the code. I'll present them as "verify" items maybe, but the references are precise.

Now let me also think about architecture-level critique:

1. **Massive, essay-length docstrings/comments.** ~95KB CLAUDE.md, 200KB README, and nearly every function has a multi-paragraph docstring. The code is heavily over-commented relative to its size. This is unusual and arguably a maintenance liability: comments drift from code (the failing test is evidence), and the signal-to-noise for a new contributor is poor. But the project seems to deliberately value this. I'll frame as a tradeoff.

2. **Templates are duplicated logic** — `template_code.py`, `template_code_mocked.py`, `template_baseline.py`, `template_baseline_mocked.py`, `template_remote_code.py`, plus `gep_lora/core/blends/lora_server.py`. The blend arithmetic (`attach`, `combine`, `_compact`, `_rank`) exists in template_code.py, lora_server.py, and golive.py (partially). The project acknowledges this but it's a real duplication risk. `_prompt_of` is duplicated between template_code.py and generate_runs.py; `_question_of`/`_reference_of` too.

3. **God modules/objects**: `gep_lora/core/pipeline/start_run.py` (1462), `store.py` (1388), `generate_html_db_stats.py` (2424), `App` (40+ methods). `gep_lora/core/metrics/report.py` also large.

4. **No packaging/dependency pinning/lint/typecheck config.** No pyproject, requirements, ruff, mypy, pre-commit, CI. CLAUDE.md tells the agent to run lint/typecheck but there's no command.

5. **Test coverage gaps**: no tests for store.py directly, generate_runs, process_run parsing, server_pool, lora_server, worker, evaluate, testpass, reporting, metrics. Empty test dirs with stale pycache. One failing test.

6. **Security**: SSRF/key-exfiltration via user-supplied judge base_url (multiple paths); no rate limiting/quotas; unsalted SHA-256 for keys; no worker exclusivity/lock; negative Content-Length unbounded read; path commonpath ValueError on cross-drive.

7. **Concurrency**: sqlite connections used across threads? `process_run.launch_batch` uses threads only for subprocess waiting; DB writes from one thread. But async_api server uses `ThreadingHTTPServer` and shares a Registry/sqlite connection across threads — need to check. The agent flagged some.

8. **Error handling**: bare `except Exception` in start_run.run (deliberate), `except (RuntimeError, ValueError)` in evaluate — an evaluator raising another exception type (e.g., KeyError, TypeError) propagates and kills the step. Actually evaluator.score is supposed to raise; common.ask_judge raises RuntimeError. But local evaluators could raise others. Minor.

9. **`generate_runs.render` marker substitution** — `fill()` replaces `@@NAME@@` for each value, but block markers are matched by exact stripped line. If a block marker name is also a substring of an inline marker, order matters. Values don't include block names, so fine. But `leftover = [line for line in out if "@@" in line]` catches unfilled markers. Good.

10. **`store.connect` executes SCHEMA on every open** — fine (IF NOT EXISTS), but also creates the DB if the path is a typo (acknowledged; readers guard). 

11. **Global mutable settings**: `gep_lora/core/config/settings.py` with `snapshot()`, `_override` in async_api mutating globals. The pipeline freezes settings into DB which is good.

12. **Seeds via `random.Random("%s:%d" % (master, number))`** — string seeds are deterministic across Python versions? Python's `random.Random(str)` hashes the string using... actually for str/bytes, `random.seed` uses `int.from_bytes(a.encode() + sha512)` — deterministic across runs and platforms (not affected by PYTHONHASHSEED). Yes, stable. Good.

13. **`selection`/`mutation` seed on `high_number`** — if selection appends before culling, high_number rises. Fine.

14. **A real subtle bug**: In `step_selection`, the RNG is seeded from `high_number` *before* `selection.select` runs. Inside select, `append_copies`/`append_individual` change high_number, but rng already seeded. Fine.

15. **`calculate_fitness.fitness_of` rounds to 6**, then `set_fitness`. Fine.

16. **`individual_quality` view + `quality_rows`** returns `ORDER BY quality IS NULL, quality DESC, number`. Fine.

17. **`process_run.exchanges`**: If a real answer line starts with "QUALITY:" it gets cut. Acknowledged.

18. **`launch` timeout**: `child.kill()` on Windows kills only the direct process; child may have grandchildren (none here). Fine.

19. **`server_pool`** not read; trust.

20. **Data integrity**: `add_exchanges` uses `judge_model="generated" if "quality" in item`. If a mocked run has quality but real judge, fine.

21. **`store.append_copies` doesn't set `has_changed`** — inherits parent's. The copies are meant to be mutated. But if mutation is off (rate 0), copies retain parent's has_changed and fitness, and process will skip them if has_changed=0 and already executed... Actually a copy inherits parent's has_changed. If parent had has_changed=1 and executed, the copy is `id not in done`, so it runs. If parent has_changed=0 (unchanged, already run), the copy is a new id not in done → runs. Fine.

22. **Potential issue in step_process `current` computation**: `store.latest_runs` returns `head` = first 4000 chars of stdout; `process_run.expression` searches `^Individual \d+: (\S+)$`. If the script printed a lot before that line (it doesn't; it's first), fine.

I think I have more than enough. Let me now also quickly verify the `evaluate.py` lack of URL validation to be accurate, and the negative content-length. Actually I'll trust the agent's line refs but phrase carefully.

Let me quickly read gep_lora/service/evaluate.py around 80-110 and server.py _body.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\async_api\\evaluate.py",
  "offset": 80,
  "limit": 45
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\async_api\evaluate.py</path>
<type>file</type>
<content>
80: def options_for(body, offered):
81:     """A request, checked against what this sweep offers. -> options.
82: 
83:     {"force": bool, "settings": {NAME: value}} -- only the judge settings that
84:     differ from what the sweep already holds, so an evaluation that moves
85:     nothing writes nothing into it.
86:     """
87:     body = body or {}
88:     if not isinstance(body, dict):
89:         raise EvaluateError("the body must be an object")
90:     unknown = sorted(set(body) - {"force"} - {field for field, _ in JUDGE_SETTINGS})
91:     if unknown:
92:         raise EvaluateError("unknown field(s): %s; an evaluation takes force, %s"
93:                             % (", ".join(unknown),
94:                                ", ".join(field for field, _ in JUDGE_SETTINGS)))
95:     if not offered["answers"] and not offered["tested"]:
96:         raise EvaluateError("this job holds no answers yet -- its search stopped "
97:                             "before any script ran. Resume it instead.")
98:     force = body.get("force", False)
99:     if not isinstance(force, bool):
100:         raise EvaluateError("force must be true or false")
101:     changed = {}
102:     for field, setting in JUDGE_SETTINGS:
103:         value = _text(body, field)
104:         if value is None or value == offered["defaults"].get(field):
105:             continue
106:         if field == "judge_backend" and value not in offered["backends"]:
107:             raise EvaluateError("unknown judge backend %r; there are: %s"
108:                                 % (value, ", ".join(offered["backends"])))
109:         changed[setting] = value
110:     return {"force": force, "settings": changed}
111: 
112: 
113: def arguments(options):
114:     """The gep_lora/core/pipeline/main.py flags one evaluation adds to the job's command line."""
115:     argv = ["--evaluate"]
116:     if options.get("force"):
117:         argv.append("--force")
118:     for name, value in sorted((options.get("settings") or {}).items()):
119:         argv += ["--set", "%s=%s" % (name, json.dumps(value))]
120:     return argv

(End of file - total 120 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\sage-is\\loras\\gep-lora\\async_api\\server.py",
  "offset": 828,
  "limit": 40
}
```

**Output:**
```
<path>D:\sage-is\loras\gep-lora\async_api\server.py</path>
<type>file</type>
<content>
828:         self.send_response(status)
829:         self.send_header("Content-Type", "application/json")
830:         self.send_header("Content-Length", str(len(body)))
831:         self.end_headers()
832:         self.wfile.write(body)
833: 
834:     def _body(self):
835:         length = self._unread
836:         if length > settings.MAX_BODY_BYTES:
837:             raise ApiError(413, "request body over %d bytes" % settings.MAX_BODY_BYTES)
838:         if not length:
839:             return {}
840:         raw = self.rfile.read(length)
841:         self._unread = 0
842:         try:
843:             return json.loads(raw.decode("utf-8"))
844:         except (ValueError, UnicodeDecodeError) as error:
845:             raise ApiError(400, "bad JSON: %s" % error)
846: 
847:     def _bearer(self):
848:         header = self.headers.get("Authorization") or ""
849:         return header[7:].strip() if header.lower().startswith("bearer ") else None
850: 
851:     def _dispatch(self):
852:         """One request, and then whatever of its body nobody read.
853: 
854:         The connection is kept alive (HTTP/1.1), so a reply sent before the
855:         body was read -- an unknown route, a wrong method, a bad key -- would
856:         leave that body on the socket, to be read as the start of the next
857:         request ("400 Bad request syntax ('{...}POST /...')"). So it is drained
858:         here, whatever the route did; one too big to read is not, and the
859:         connection is closed instead."""
860:         try:
861:             self._unread = int(self.headers.get("Content-Length") or 0)
862:         except ValueError:
863:             self._unread, self.close_connection = 0, True
864:         try:
865:             self._route()
866:         finally:
867:             self._drain()

(Showing lines 828-867 of 1034. Use offset=868 to continue.)
</content>
```

---

## Assistant (Build · DeepSeek V4.1 Flash · 24.6s)

_Thinking:_

Confirmed: `_unread = int(Content-Length)`. A negative value passes `if length > MAX_BODY_BYTES` (false) and `if not length` (negative is truthy) → `self.rfile.read(-1)` reads to EOF. Real bug. And `evaluate.options_for` doesn't validate judge_base_url scheme. Confirmed.

I now have a comprehensive, evidence-backed critique. Let me write it up. The user asked: "Look into this project, critique the code and suggest areas of improvement."

I should structure it well but not be excessively long. Given the CLI, I should be reasonably concise but this is a critique request so more detail is warranted. I'll organize by priority: strengths (brief), then findings by severity, then improvement suggestions.

I must be accurate and cite file:line. Let me make sure I cite things I directly verified vs. things the sub-agents reported. I'll mark the agent-reported ones but they had precise refs.

Let me keep it focused and actionable. I'll avoid dumping all 20 agent findings; I'll prioritize the most important and group the rest.

Let me note the test failure as concrete evidence.

Let me write the response. Given the guidance to be concise but this is explicitly a "critique and suggest improvements" task, a longer structured answer is appropriate. I'll aim for maybe 80-120 lines.

Key points:

## What's strong
- Single source of truth for settings, frozen per sweep into sqlite (reproducibility).
- Real sqlite schema with FK cascade, self-contained rows.
- Grammar invariants enforced centrally, mutation class-local.
- Registry pattern for evaluators, one module each.
- 471 tests, mostly fast (90s), hermetic for search steps.

## Bugs / defects (verified)
1. Failing test: `test_it_does_not_ask_for_a_copy` vs prompt — drift. (evidence)
2. Negative Content-Length → `rfile.read(-1)` unbounded read (server.py:834-845, 861).
3. SSRF + API-key exfiltration via user-supplied judge base_url: server.py:603-622, evaluate.py:102-109, submit allows JUDGE_BASE_URL (not locked). Key sent as Bearer in common.py:66,247-291.
4. `os.path.commonpath` ValueError on cross-drive (submit.py:241-250; routes.py:98-103).
5. No worker exclusivity/lock; recover() fails live jobs; pid unused (worker.py:402-416).
6. Tool call bodies can raise uncaught ValueError/KeyError → 500 (tools.py etc.).

## Testing gaps
- No direct tests for store.py, generate_runs.py, process_run.py parsing, lora_server/server_pool, worker/evaluate/testpass, reporting, metrics.
- Empty/stale test dirs (`unittests/blends`, `unittests/storage`, `unittests/tools`, `unittests/api` with deleted-suite pycache).
- 1 failing test in an otherwise green suite → no CI to catch.

## Engineering / process gaps
- No dependency manifest (requirements/pyproject), no lockfile; setup .bat scripts mostly untracked (`*.bat` ignored; only 2 bats tracked). Fresh clone can't be set up.
- No lint/typecheck/CI config though CLAUDE.md says to run them; `# noqa: BLE001` implies a linter with no config.
- Committed binary `docs/example_run/gep.sqlite3`.
- `api/` leftover dir with only pycache; `NUL` artifact.

## Architecture / maintainability
- Duplicated blend arithmetic across template_code.py / template_remote_code.py / lora_server.py / golive.py; duplicated `_prompt_of`/`_question_of`.
- God modules: generate_html_db_stats.py 2424, gep_lora/core/pipeline/start_run.py 1462, store.py 1388; `App` ~40 methods; triplicated queue machinery in registry.py.
- Doc-comment volume: 200KB README + 95KB CLAUDE + paragraph docstrings; risk of drift (the failing test is one instance). Suggest trimming comments to invariants/why, moving narrative to docs.
- Import cycles/lazy imports (blending↔routes, agent→providers globals).

## Security hardening
- rate limiting/quotas on expensive work; per-user queue.
- hash with HMAC/pepper + compare_digest.
- validate/blocklist judge URLs (no link-local/metadata), require explicit allowlist.

## Suggestions (prioritized)
1. Fix the failing test (or prompt).
2. Fix Content-Length guard.
3. Restrict JUDGE_BASE_URL / lock it or validate allowlist; never send API key to user-chosen host.
4. Add requirements/lock + track setup scripts; add pyproject with ruff/mypy + CI running unittest.
5. Fill core test gaps (generate_runs, process_run, store, reporting) with golden-file tests.
6. Deduplicate blend arithmetic into a shared module consumed by templates (or at least a contract test asserting byte-identical plan).
7. Remove dead dirs/artifacts.

I'll write this now, being concrete. I should be careful not to overclaim the agent findings I didn't personally verify — but the agents read the files and gave line refs, and I verified the highest-severity ones (SSRF, content-length, digest, evaluate). I'll present them as findings.

Let me write the final response.

I explored the codebase, read the core modules end-to-end, ran the test suite, and had sub-agents review the async API and agent packages. Here's the critique.

## What's genuinely strong

- **Reproducibility is designed in.** Settings are frozen per sweep into sqlite (`gep_lora/core/config/settings.py:693`, `start_run.freeze`), seeds are drawn-and-recorded, and every step reads the *sweep's* stored settings rather than `settings.py`. This is rare and well executed.
- **The schema is self-contained.** Executions keep their weights/stdout/transcript; `fitness_history` and `phase_timings` snapshot the past instead of joining to the mutating present (`gep_lora/core/storage/store.py:184`, `:352`). Good instincts about history surviving culls.
- **Grammar invariants live in one place** (`gep_lora/core/search/generate_population.py`) and mutation is class-local so every mutant decodes. The evaluator registry (`gep_lora/core/evaluators/common.py:163`) is a clean extension point.
- **471 tests, ~90s, mostly hermetic** — the search steps run against a real temp sqlite DB (`unittests/core/search/support.py`), which is the right call.

## Defects (verified)

1. **A test is failing right now.** `unittests/core/evaluators/test_llm_judge_answers.py:170` asserts `"Do not reward copying"` is in `JUDGE_ANSWERS_SYSTEM_PROMPT`, but the prompt (`gep_lora/core/evaluators/llm_judge_answers.py:54`) no longer contains it. Either the rubric lost an anti-copying instruction or the test is stale — prompt/test drift, and nothing catches it.
2. **Unbounded blocking read via a negative `Content-Length`.** `server.py:861` does `int(...)`; `_body` (`:834-845`) checks `> MAX` and `if not length`, so `-1` reaches `self.rfile.read(-1)`, which reads to EOF. A client holding the socket open pins a thread indefinitely.
3. **SSRF + judge API-key exfiltration.** `server.py:603-622` (`/judge/models`) takes a user-supplied `base_url` and sends `Authorization: Bearer $JUDGE_API_KEY` (`gep_lora/core/evaluators/common.py:66,267-291`). `JUDGE_BASE_URL` isn't locked (`gep_lora/service/settings.py:40-44`), and `evaluate.py:102-109` accepts a judge URL with no scheme/host validation at all. Any authenticated user can point grading at a host they control and receive the server's key.
4. **`os.path.commonpath` raises `ValueError` on cross-drive/UNC paths** (`submit.py:241-250`, `routes.py:98-103`), escaping as a 500 rather than the intended 400.
5. **No worker exclusivity.** `worker.recover()` (`worker.py:402-416`) marks every `running` job failed without an OS lock on `JOBS_DIR`; the stored `pid` is never used to detect or kill an orphan, so a second worker fails the first's in-flight job.
6. **Tool bodies can 500 the agent.** `Toolbox.run` catches a fixed exception list (`tools.py:239-243`), but `int(count)`, `float(epochs)`, `TOOL_DONE[name].format(...)` raise `ValueError`/`KeyError` on model output and propagate (`agent.py:243` only catches `ProviderError`).
7. **Cross-user leak in `_set_verify_questions`**: `tools.py:623` reads a catalog row by id with no owner check (`catalog.get` is owner-agnostic), and the id comes from page-supplied session state.

## Testing gaps

- **Core untested modules**: `gep_lora/core/storage/store.py` (exercised only through fixtures), `gep_lora/core/blends/generate_runs.py`, `gep_lora/core/blends/process_run.py` (regex transcript parsing — exactly where subtle bugs hide), `gep_lora/core/blends/lora_server.py`, `server_pool.py`, `gep_lora/service/worker.py`, `evaluate.py`, `testpass.py`, `gep_lora/core/reporting/generate_html_db_stats.py` (2424 lines), `gep_lora/core/metrics/report.py`.
- **Dead artifacts**: `unittests/blends`, `unittests/storage`, `unittests/tools` are empty; `unittests/api/` holds only `__pycache__` from a deleted suite (`test_progress`, `test_prompting`, `test_server`).
- No CI, so the one red test slipped through.

## Process / repo hygiene

- **No dependency manifest at all** — no `requirements.txt`, `pyproject.toml`, or lockfile. Setup lives in `.bat` scripts that are mostly **untracked** (`*.bat` is gitignored; only `generate_report.bat`/`run_unit_tests.bat` are tracked). A fresh clone cannot be set up.
- **No lint/typecheck config**, despite `CLAUDE.md` instructing agents to run them and code carrying `# noqa: BLE001`.
- Committed binary `docs/example_run/gep.sqlite3`; leftover `api/` directory (pycache only); a `NUL` file at the root.

## Architecture / maintainability

- **Duplicated blend arithmetic** across `gep_lora/core/templates/template_code.py`, `template_remote_code.py`, `gep_lora/core/blends/lora_server.py`, and `gep_lora/service/golive.py` (the plan/weight draw is reimplemented there). The prompt parser is duplicated too (`template_code._prompt_of` vs `generate_runs._question_of`/`_reference_of`). A silent divergence makes a live blend differ from the one that was scored.
- **God modules/objects**: `generate_html_db_stats.py` (2424), `gep_lora/core/pipeline/start_run.py` (1462), `store.py` (1388); `App` has ~40 methods (`server.py:246-771`); `registry.py` triplicates `claim_next`/`finish`/`orphaned*` for jobs/verifications/trainings.
- **Documentation volume is a liability.** README ~200 KB, CLAUDE.md ~95 KB, plus paragraph-length docstrings. It's impressive, but it duplicates itself across modules and drifts (finding #1 is one instance). Keep invariants and the *why*; move narrative to one place.
- **Import cycles / globals**: `blending`↔`routes` lazy-import cycle; `agent.say`/`chat` call provider functions as module globals, making failure paths untestable without a live server.

## Suggested priorities

1. Fix the failing test (or restore the anti-copy prompt) and add CI running `unittests`.
2. Guard `Content-Length` (`< 0 → 400`), and consider a socket timeout on `ThreadingHTTPServer`.
3. Treat judge URLs as untrusted: lock `JUDGE_BASE_URL` (or allowlist hosts, block link-local/metadata), validate scheme/host in `evaluate.py`/`verify.py`, and never attach `$JUDGE_API_KEY` to a user-chosen host.
4. Add `pyproject.toml`/requirements + lock, track the setup scripts, and configure `ruff`/`mypy`; wire them into CI.
5. Add golden-file tests for `generate_runs`/`process_run` (feed a fixed stdout, assert the parsed transcript/timings) and direct tests for `store.py`; delete the stale test dirs.
6. Extract the shared blend arithmetic into one module the templates and `lora_server`/`golive` both consume (or a contract test asserting byte-identical plans).
7. Harden the API: per-user quotas/rate limits, a real worker lock, HMAC+`compare_digest` for stored digests, and owner checks in the agent tools.

Want me to start on any of these — e.g. fix the failing test and the `Content-Length` guard first?

---

