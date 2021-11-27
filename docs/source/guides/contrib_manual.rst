============================
How To Contribute (Extended)
============================

Do you consider contributing to our project? That's great! We welcome all kinds of contributions - whether you discovered a bug, want to support the project by helping other users, code new features or just give some general feedback.

.. TIP::

   If you have some experience contributing to open source projects, check out our Contributing file on GitHub to get you started. This guide is intended for new users that would like to contribute to our project but would prefer a more in-depth explanation of all the steps necessary to do so.

We're firm believers of open source and want to create an inclusive environment around our project where everybody is welcome to join! In this spirit, you don't need a lot of experience in Python to contribute, and we try hard to provide all necessary information to anybody not familiar with all this DevOps stuff. This guide is an extended manual that aims to cover all topics needed to get you started developing in our project! The covered topics include

* installing the repository in *development-mode*
* setting up your dev-environment
* conventions and best-practices we use
* necessary steps for a successful pull request (PR)
* some explanations for why this is so much more complicated than scripting.


Preliminaries
=============

There are a few pre-requisites for using this guide:

* An account on `GitHub <https://github.com/>`_ where we host all the source code.
* Both `python ~ 3.8` and `git` installed on your machine. If that's not yet the case, install python from the `official page <https://www.python.org/downloads/>`_ (or check out `pyenv <https://github.com/pyenv/pyenv>`_) and `git <https://git-scm.com/>`_.

.. NOTE::

    This guide was designed and tested on Windows. If you use a different OS, the basics still apply, although the specific steps during setup might differ. We try to include sources for all these steps so that you can check for yourself if there are OS-dependent changes on how to proceed.


Installation in *dev* mode
==========================

Now, lets install our repository in *dev*-mode. This installation differs from *production*-mode. The latter describes the ready-to-use version of the package as you would install it e.g. from PyPI. *Dev*-mode, on the contrary, refers to the direct copy of the repository as you would find it on GitHub, including tools used for developing, testing, quality assurance and all (public) branches where new features are developed. This mode has more dependencies which change regularly as we develop the next release, i.e., a snapshot of a publishable version of the code. Dependency management (and packaging) used to be inconvenient in Python, yet it is important that every contributor works on the same environment when collaborating on code. Luckily, there's a tool called *poetry* which simplifies a lot. Thus, the first step is to install poetry!

Poetry requires a system-wide installation that's different on Windows and MacOS. The full installation instructions can be found on `the official page <https://python-poetry.org/docs/master/#installation>`_. The quick four-step-version goes:

1. Open Windows PowerShell
2. Copy, paste and execute

.. code-block::

    (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py -UseBasicParsing).Content | python -


3. The installer will tell you the path of the installed executable. Add the path to your system ``PATH`` if that's not done automatically.

4. Re-open PowerShell and execute `poetry --version` to check if the installation was successful

With poetry and git, getting the right environment is now rather easy. First, you need to get a copy of the original repository. To do so, follow these steps:

1. Go to the `repository page <https://github.com/floodlight-sports/floodlight>`_
2. Hit the **Fork** Button on the top right of the page. This will create a personal blueprint in your github account. Compared to the base repository, you have the permission to manage this repository in whatever way you like
3. Clone the repository to your local machine as usual.

You've now got your own "version" of the original repository on your machine. The last step is to install all the necessary dependencies. Go the repo's directory and just run `poetry install` (e.g. in git bash)! Poetry will then create a `virtualenv` and install all necessary dependencies. This is basically everything you need to start contributing. However, we follow a number of conventions to ensure code quality. Make sure you know and follow these conventions so that your code fits nicely into the existing codebase!


Conventions
===========

These are:

1. Codestyle
    - [PEP8](https://www.python.org/dev/peps/pep-0008/) and the [Zen of Python](https://www.python.org/dev/peps/pep-0020/).
    - [Typing](https://docs.python.org/3/library/typing.html)
    - [Docstrings](https://www.python.org/dev/peps/pep-0257/) in [numpy-style](https://numpydoc.readthedocs.io/en/latest/format.html) (as in this [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html))
2. DevOps
    - Structured commit messages with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
    - The [git-flow](https://nvie.com/posts/a-successful-git-branching-model/) branching model
    - Semantic Versioning ([SemVer](https://semver.org/)) for versioning


Local workflows
===============

Most of these conventions are enforced through the contributing workflow
(fork - clone - edit - pull request) as well as automatically with GitHub Actions used for
continuous integration purposes. However, you may want to ensure a local dev environment that
actively facilitates these conventions. There are a number of tools you can use to do so:

#### Pre-Commit Hooks

You can install pre-defined pre-commit hooks by running:

````
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg
````

These hooks will automatically get activated whenever you commit any code, and check for code style
(via black and flake8) as well as commit message structure. You can also activate each of these tools
manually by running the following commands (see the respective docs for full intros):

*Note: you need to start every command with `poetry run` if executables are not in your `PATH`*

- re-format all code with black: `black .`
- run linter: `flake8`
- dummy check all pre-commit hooks: `pre-commit run --all-files`
- update hooks: `pre-commit autoupdate`
- check a commit message: `cz check -m "my commit message"`


#### PyCharm

Additionally, if you use an IDE like PyCharm, you can set up your favorite tool to help you right
during coding. For example:

1. Add new Interpreter and point to python.exe in poetry-created env
2. [Integrate black](https://black.readthedocs.io/en/stable/integrations/editors.html) (you could do the same with flake8)
3. Configure Inspections -> PEP8 checking
4. Setting > Tools > Python Integrated Tools: Set default tester and docstring format



Global Workflows
================

Once you have made your fork and clone of the original repository, there are three copies that are of interest:

- the original repostiry, hereafter called `base` or `upstream`
- your fork that's stored on GitHub (`origin`)
- the local clone on your machine (`local`)

Up to this point, you're set up so that you can develop on `local`. The remaining question is: once you've done some work and coded that cool new feature, how do you get your changes into `base`? There are two major ways of doing so.

### Develop locally, merge remotely

This is the standard way for contributing to an open source repository without having direct write access. In a nutshell, you want to keep your `local` up to date with `base`, develop a new feature on `local`, and request to merge it into `base` once you're finished. The long story goes like this:

Remember that we follow a (slim-fit) version of the git-flow model, which gives the `main` and `develop` branch a special role. These are reserved for stable snapshots of the code (`main`) as well as (potentially unstable) checkpoints during development of a new version (`develop`). There's two implications here:

- You want to keep your local copies of these two branches up to date with the original ones to avoid merge conflicts due to missed updates
- You shouldn't work on these branches directly but use feature- or hotfix-branches for your work that branch from and merge into `develop`

If you add your own feature branch, there's now three repositories and three branches flying around. This might be puzzling at first sight, maybe take a moment and try to sort these out. On second sight, however, the GitHub-workflow and git-flow model are great teamplayers. There's a one-way road opening up that goes like this:

`base:main` /`base:develop` > updates > `local:main`/`local:develop` > branches > `local:my_feat_branch`

That's pretty much half of the cycle that starts at `base` and ends at your local feature branch. The other half goes in a different direction as you're lacking write access to push your changes up the road where the original code came from. Here, you need to take a little detour over `origin` - your GitHub copy of `base`:

`local:my_feat_branch` > pushes > `origin:my_feat_branch` > merge > `base:develop`

Again, you would need write access to `base` to perform the merge in the last step by yourself. Instead, the final step of contributing your code is handled by GitHubs **Pull Request (PR)**. Essentally, you use GitHub to explain/present your work, show that it passes all the workflows triggered by GitHub Actions and ask the maintainer to merge your changes.

So much of the theory, let's see how one can perform all these steps in practice:

1. It's important to keep your `local` up to date with `base`, so that your contribution integrates smoothly with the current version instead of relying on code that's a few commits behind. To this end, you may add `base` as an additional remote location so that from now on you can pull new commits directly from there:

   ```git
   git remote add upstream https://github.com/floodlight-sports/floodlight
   git fetch upstream
   ```

2. As you never push to `origin:develop` or `origin:main` anyways, you can let them track `base:develop` and `base:main` instead. For `develop` that's done by:

   ```git
   git checkout develop
   git branch -u upstream/develop
   ```

   Same goes for `main`.

3. Don't use `main` or `develop` for your development directly, rather keep them in sync with the equivalent branches in `base` by hitting

   ```git
   git pull
   ```

   on the respective branch.

4. For your new feature, create a new branch from the latest version of the code:

   ```git
   git checkout develop
   git checkout -b my_feat_branch
   ```

5. Code and commit on this branch as you would normally do.

6. Once you're finished, make sure you haven't missed any updates on `base` while you were coding:

   ```git
   git checkout develop
   git pull
   git checkout my_feat_branch
   git rebase develop
   ```

7. Push the changes to your GitHub fork:

   ```git
   git push -u origin
   ```

8. Go to the [repository page](https://github.com/floodlight-sports/floodlight) and do a PR. Make sure you ask to merge your changes from `origin:my_feat_branch` into `base:develop`.



## Resources

https://git-scm.com/book/de/v2/GitHub-Mitwirken-an-einem-Projekt

