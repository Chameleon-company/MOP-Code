# Development guide for AI + IoT team
*Updated in T2 2025 by Dac Cong Nguyen.*

## Repository structure
Since T1 2025, we have started organising our repository as the following structure:
```
MOP-Code/artificial-intelligence
  - <Trimester-name>/
    - <Sub-team-name>/
      - <Use-case-name>/
```
In each trimester folder, only **active** use cases (a use case is the smallest unit of the project, each use case should be considered as a distinct project) are presented. However, the code for the use cases are **not necessary** in their dedicated folders. Depending on the origin of the use case, we might decide where to put the code of that use case based on the rule:
- In case the use case already existed in the previous trimesters, the current trimester is to further develop it: The main code should still be in the existing path, with a new folder for that use case in the current trimester folder noting the changes and pointing the existing code base path.
- In case the use case is new: The code should be located in the current trimester folder.

The idea for this structure is to ensure that all the active use cases in a trimester are presented, while avoiding the duplication of the same code (by copying the same code base over different trimesters).

## Contribution guide
Each use case team should work in their own branch. Every contribution during the trimester should be first merged into that branch, and finally the use case branch will be merged into `master`.
```
At the start of the trimester - Create a branch to work for the use case
# From master
git checkout -b your_use_case_branch

During the trimester - Create branches to develop new features
# From your_use_case_branch
git checkout -b your_feature_branch

Regularly commit and merge to your_use_case_branch
- Create pull requests to merge code from your_feature_branch to your_use_case_branch

At the end of the trimester
- Create pull request to merge code from your_use_case_branch to master
```

Besides the code, each use case also needs a clear documentation, which is presented in the README file in that use case folder. It should effectively shows the use case overview, environment setup, reproduction guidelines, current status, and the ideas to further develop (if any) of the use case.

## Tasks for GitHub administrator
1. Onboard the members to the repository.
2. Create trimester folder and the active use case folders.
3. Provide contribution guidelines for team members.
4. Participate in code finalisation of the repository at the end of the trimester.