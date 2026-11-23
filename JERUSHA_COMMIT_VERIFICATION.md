# AcousticSpace — Manual Commit Verification Checklist

This checklist provides a small, non-functional documentation update for testing GitHub commits from a collaborator account. It does not modify application code, model weights, datasets, or evaluation results.

## Verify the GitHub upload

1. Sign in to GitHub using the intended collaborator account.
2. Open the `AkashTheSciGuy/AcousticSpace` repository and select the existing `backend` branch.
3. Upload this file to the repository's `docs/` directory.
4. Commit with a descriptive message, for example:
   `docs: add collaborator commit verification checklist for AcousticSpace`
5. Open the commit details and confirm the **author** is the intended account.
6. Check the repository's Actions tab for CI status, if the workflow runs on the branch.
7. Compare the repository's branch and commit history to ensure the change is on `backend`, not an unintended branch.

## What a successful test proves

- The account has permission to commit to the chosen branch.
- The uploaded file appears in the repository history.
- GitHub records the commit's author and timestamp.

## What this test does not prove

A commit appearing in the repository does **not** guarantee it will appear on the account's contribution calendar or count toward any internship dashboard. Those systems may use additional conditions, such as email attribution, branch rules, reporting windows, or their own validation criteria.

This document is only a commit-workflow check. It does not claim improved deepfake-detection, AST, RIR, breath-analysis, or tamper-localization performance.
