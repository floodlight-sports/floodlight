=====================
About this Compendium
=====================

.. TIP::
   If you would like to use our package without worrying about all the fiddly details, you should check out our guides and API references. However, if you want to take a deep dive into sports data analysis and the engine room of this package, look no further!


Match and performance data from all kinds of sports has seen a remarkable upsurge over the past decade. Better technology and professionalization has led clubs across countries and leagues to collect massive amounts of data and monitor almost every performance related aspect within their organization. Regardless of your background - professional sports, science, industry, hobby developer or fan - sports data analysis is exciting and an awesome road to be travelling on these days!

Unfortunately, this road as of today is full of pitfalls. Proprietary data, dozens of different data providers (each having their own specific data format), a lack of definitions (what's a "pass" or "shot" anyway?), little (but increasing!) open source algorithms, a full plate of nice-to-look-at but sometimes not-so-validated models and so on and so forth.

The *floodlight* package aims to provide a high-level framework that tackles some of these issues in team sports. In a nutshell, the basic idea is to create a set of streamlined data structures and standard functionality to make your life working with sports data is a lot easier. While we're aiming at an intuitive and easy-to-use high-level interface, there's still plenty of complexity unfolding under the hood. Creating basic, flexible data structures that can hold almost any information about sports play - independent of the data provider - is a challenge that asks for a rather technical reflection of the matter.

And that's where this compendium comes into play. The idea of this read is to provide you with an in-depth explanation of core concepts, challenges and caveats when working with team sports data. We additionally explain how each of these issues has influenced design choices, how certain problems are tackled in our implementation and which conventions we've established to tackle the complexity inherent to the task. We hope this document is of help if you want to know more about sports data in general or as an in-depth guide to the package if you want to start developing with us!

Let's get started!
