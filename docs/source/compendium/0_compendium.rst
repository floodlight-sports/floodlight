=====================
About this Compendium
=====================

.. TIP::
   If you would like to use this package without worrying about all the fiddly details, check out the guides and API references. However, if you want to take a deep dive into sports data analysis and the engine room of this package, look no further!


Match and performance data from all kinds of sports has seen a remarkable upsurge over the past decade. Better technology and professionalization has led clubs across countries and leagues to collect massive amounts of data and monitor almost every performance related aspect within their organization. Regardless of your background - professional sports, science, industry, hobby developer or fan - sports data analysis is exciting!

Unfortunately, sports data analysis also comes with a range of pitfalls. Proprietary data, dozens of different data providers (each having their own specific data format), a lack of definitions (what's a "pass" or "shot" anyway?), little (but increasing!) open source algorithms, a huge selection of proposed models with sometimes uncertain validity and so on and so forth.

The *floodlight* package aims to provide a high-level framework that tries to help with some of these issues in team sports. In a nutshell, the basic idea is to create a set of streamlined data structures and standard functionality to make your life working with sports data a lot easier. While we're aiming at an intuitive and easy-to-use high-level interface, there's still plenty of complexity unfolding under the hood. Creating basic, flexible data structures that can hold almost any information about sports play - independent of the data provider - is a challenge that asks for a rather technical reflection of the matter.

And that's where this compendium comes into play. The idea of this read is to provide you with an in-depth discussion of core concepts, challenges and caveats when working with team sports data experienced by the core development team over the past years. We additionally demonstrate how each of these issues has influenced design choices, how certain problems are tackled in our implementation and which conventions we've established in an attempt to tackle the complexity inherent to the task. We hope this document is of help if you want to know more about sports data in general or as an in-depth guide to the package if you want to start developing with us!

Let's get started!
