
# 1st progress update

## Change log

<!-- @TODO @Hevagog place your changes here -->


---

### GUI & Visualization

Apart from a command line interface (CLI) "visualization" (suited for training the agents), we have also implemented a graphical user interface (GUI) that allows us to visualize the environment and the agents' actions. The GUI is presented in [@fig:pr1-gui].

### Environment map generation

We have implemented a simple map generation tool that allows us to create a *map* file based real-world data from OSM.
It generates obstacles for all the buildings in the area and places agents in given locations.
Moreover, it generates a `png` file with a depiction of the area to be used as a background for the GUI.
Figure [-@fig:pr1-gui] presents a screenshot of the GUI with a map of the AGH University of Science and Technology in Kraków, Poland.

---

\vspace{1em}

![GUI of the application](img/pr-1/gui.png){#fig:pr1-gui height=90%}
