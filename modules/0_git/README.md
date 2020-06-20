# Git, GitHub and Travis CI

This module will show you how
* Fork repository and open a Pull Request (PR)
* Create new branches using `git` and push the changes by commits
* Work with Contiguous Integration (CI) system [Travis CI](https://travis-ci.org/)

## Details
* Create a GitHub account
* Click fork button to create a copy of origin repository

    ![](../../data/git_fork.png)

* Clone forked repository (`USERNAME` is your account name) and navigate to new folder.

    ```bash
    git clone https://github.com/USERNAME/openvino_practice

    cd openvino_practice
    ```
* Create a new branch with name `practice_git` or different

    ```bash
    git checkout -b practice_git
    ```

* Make some changes

    ```patch
    --- a/modules/0_git/test/main.cpp
    +++ b/modules/0_git/test/main.cpp
    @@ -7,7 +7,7 @@
     TEST(git, say_hello) {
         myspace::A a;
         EXPECT_EQ(myspace::func(a), "Hello, Nizhny!");
    -    EXPECT_EQ(func(a), "Hello, World!");
    +    EXPECT_EQ(func(a), "Hello, Nizhny!");
     }
    ```

* Create a commit

    ```bash
    git add modules/0_git/test/main.cpp
    git commit -m "initial commit"
    ```

* Push local branch to remote repository

    ```bash
    git push origin practice_git
    ```

* TODO: open a PR

* Wait for validation results:
    * ![](../../data/git_ci_progress.png) - tests in progress
    * ![](../../data/git_ci_failed.png) - tests failed
    * ![](../../data/git_ci_passed.png) - tests passed
