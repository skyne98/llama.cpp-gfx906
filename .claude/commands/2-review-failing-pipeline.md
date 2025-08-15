Currently this branch is failing the pipeline.

Please review the PR and associated pipeline and fix the issues.

Use the following commands to review the pipeline:

### How to get the PR number for current branch
```
gh pr status
```

### How to get run ID of the failed job (will need to filter by branch)
```
gh run list --branch <branch-name>
```

### How to get logs of the failed job in the pipeline
```
gh run view <run-id> --log-failed
```
