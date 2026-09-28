# Main branch integrity policy

VertiMosaic treats `main` as the source of published research evidence. The intended GitHub repository settings are:

- require a pull request before merging;
- require at least one approving review when another maintainer is available;
- require review from Code Owners for changes to owned security/release surfaces;
- dismiss stale approvals after new commits;
- require conversation resolution before merging;
- require branches to be up to date before merging;
- require successful status checks for CI, Security, CodeQL, Portability, Optional crypto backends, and provider-backed linked benchmark checks when they are applicable;
- block force pushes and branch deletion;
- do not allow required checks to be bypassed for routine development;
- restrict release tags to commits that are already current `main`.

The repository also contains `.github/CODEOWNERS` and a guarded `release/v*` workflow so these controls remain documented and testable even when repository-administration APIs are unavailable to an automation client.

Branch protection/rulesets are GitHub repository settings, not properties that can be guaranteed by source files alone. Verify the live GitHub rule before relying on this document as evidence that protection is enabled.
