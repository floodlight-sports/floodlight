======
Design
======


At this point, let's summarize a few design principles that we decided build our package around. These are problem-specific and on top of any general software design principles. During implementation, we've naturally encountered questions on how to solve *this* or how to incorporate *that* special case. Often then, half a dozen possible solutions come to mind, none of which appears to be clearly right or wrong. If you've decided to contribute and find yourself in this situation, these principles hopefully come in handy.

The scope of this package is another reason we explicitly formalized these principles. Designing a high-level framework bears the natural risk of mutating into a jack of all trades that does everything, but nothing really well. There are plenty of use cases our package might come in handy, and even more possible ways to extend it. Yet, each of this cases could require a different implementation focus, and possibly conflicting ones. To keep things from diverging, these principles should also roughly narrow down the perspective we aim to take when adding functionality.

1. **Data Analytics Focus**

    All data structures and functionality should be implemented with a clear focus on scientific data analytics. We attempt to optimize core objects and manipulation functions so that data processing becomes intuitive and easy. If a trade-off has to be made, we prioritize data handling over (external) API compatibility, database friendliness, or use-case specific requirements. Performance is also an issue we keep in mind.

    On an implementation level, we make extensive use of data analysis packages such as *numpy*, *pandas*, or *matplotlib*. We let the experts do what they're the best at, and gratefully use their great tools. Many of our core data objects are container classes wrapping ``ndarray``\s or ``DataFrame``\s. This way, we can add our own flavour and add needed functionaliy, while leveraging and interfacing their rich functionality such as views, indexing, and vectorization.

2. **Sports Independence**

    The basics of processing sports data apply to many sports across the board. Most spatial or temporal transforms, data manipulation techniques or performance metrics apply irrespectively of whether there are eleven players chasing a ball, or five a puck. We intend to provide essential data structures and functionality for all team invasion games, such as football, basketball, handball, hockey, and so on.

    We are aware that data analysis on football draws by far the major portion of attention. Submodules that solve a sport-specific problems are certainly welcome, but please double check whether your contribution can be further abstracted to apply to all team sports. If not, that's okay, but make sure to mark it as sport-specific!

3. **Provider Independence**

    Our :doc:`io <../modules/io/io>` submodule handles all necessary heavy lifting to parse provider data into core objects. They are intended to provide a clear interface to data loading, and anything that is *not* related to parsing should be implemented completely provider independent.

    Obviously, there is functionality that requires certain data flavours specific to individual providers. We welcome these additions, and have some included already. However, they should be based off core objects and enforce necessary conditions to match provider-specific requirements.

4. **Soft Constraints Enforcement**

    To fit as many data flavours and use cases as possible, we design all core objects with minimal requirements on the data. All the conventions we include are the ones we found to be absolutely essential to abstract the data and provide a common ground to operate on. However, this does not prevent us from introducing "soft conventions". Any constraints on data or functionality that exceeds the basics is included not as an absolute requirement, but rather as an prerequisite for certain functionality.

    For example, event-related information comes in various scopes and shapes, and is handled via columns within the :doc:`Events <../modules/core/events>` objects ``DataFrame``. You can put anything you like into these ``DataFrame``\s, and there is no limitation on column names. However, there are two *essential columns* ("eID" and "gameclock") that are the absolute minimum to describe and locate events (hard constraints), and are necessary for construction.

    To handle event properties beyond those two descriptors, we do not enforce any definitions or mappings. Instead, we provide a short list of *protected columns* that are unambiguous to standardize, such as "at_x" and "at_y" to include event locations. You may use the object as you like, but any method that requires this information then checks if the respective column names are available, and throws and error if not (soft constraints).

5. **Sensible Defaults**

    We aim to design any functionality as little opinionated as possible. There should be a general way to do things, with neutral outputs that do not take a subjective perspective.

    However, we are fully aware that sports data processing is full of choices. Many algorithms or standard procedures can be done this way or that way, which require selecting a default behavior. In these cases, we intend to choose a sensible default that is independent of any data flavours or personal preferences. In case multiple options are available, we inform the user about our choice and include handling processing alternatives via function parameters.

    We also try to not exclude any potential personal preferences or use cases by requirements that are to strict, or function calls that are narrower in their scope as their name and description promises.

6. **Intuitive High-Level Interfaces**

    Data processing is complex, and we aim to tackle this complexity with our approach. Thus, all low-level interfaces may bear a complexity that is familiar to users with a data science background. However, we also acknowledge that the package may be used by less experienced users. These users may not have the experience to establish all steps of a full processing pipelines all the way from raw data to final results.

    In this case, our framework might help set up an environment that takes away the worries of data handling and lets the user focus on analysis. To enable such an approach, all high-level functionality should have intuitive and easy to use interfaces.
