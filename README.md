# Autonomous Systems Design Laboratory

[![Fork](https://img.shields.io/badge/Fork-me-blue)](https://gitlab.hs-esslingen.de/anbait07/asd-docker-template/-/forks/new)

This is a template repository for the Autonomous Systems Design lab course using ROS. Start by forking this repository by hitting the `Fork me` button above.

## Development

The provided devcontainer comes with a few tools to help with development.

![Launch Simulation](doc/launch-tutorial.gif)

The above animation shows how to compile the project and start a simulation from inside vscode.

1. First open the noVNC page in a browser. Run `>Ports: Focus on Ports View` from the command palette and hit the preview button on Local Address `localhost:6080`. This will open a browser window inside vscode, connecting to noVNC inside the devcontainer (password: `vscode`).
2. Open the terminal panel and run the following commands:

    ```sh
    catkin_make -C /workspace # build the project
    source /workspace/devel/setup.bash # load project into ROS environment
    roslaunch car_demo demo_keyboard.launch # launch simulation
    ```

3. After starting the simulation switch back to noVNC. There you'll see the RViz GUI. If the window is out of view, right-click the taskbar and hit `Maximize`.
