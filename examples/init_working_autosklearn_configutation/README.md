Draft folder that contains the configuration that successfully runned on Decentriq. We were provided a debug instance of the confidential computing machine, since auto-sklearn is not one of the available packages in Decentriq.

Some issues that we faced:
- instance specific errors, solved with re-logging in the web API
- timeout of "test all computations" (we can request a larget time window in the future if needed)
- unsuccessful shut-down of autosklearn inside VMs/Dockers, which causes the VM to be stuck

The overall solution was to force shutdown from inner scope, as can be seen in `training_script.py`. From what I understood, this is an issue that takes place when Decentriq is evaluating the computations and not during the actual execution. In any case we need to handle it, since "test all computations" seems to be a prerequisite for publishing a DCR.

A json file that defines the DCR and the final executed script are commited to be stored in the repo's history.
