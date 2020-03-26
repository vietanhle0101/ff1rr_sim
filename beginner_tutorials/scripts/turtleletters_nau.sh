#!/usr/bin/env bash
# Draw the letters "NAU" by a turtle

# Reset the simulation, with turtle1 at (0,0,0)
rosservice call /reset

# 1. Letter N

# 1a. Transport turtle1 to first point of the letter

# Turn off pen
# The arguments are: R, G, B, line-width, pen-off? (1 => off, 0 => on)
rosservice call /turtle1/set_pen 200 0 0 5 1

# Transport the turtle
# <COMPLETE THE INITIAL COORDINATE AND ANGLE BELOW>
rosservice call /turtle1/teleport_absolute '{x: 2, y: 3, theta: 1.57}'

# 1b. Draw the letter line by line

# Turn on pen
< CALL SERVICE TO TURN ON THE PEN >
rosservice call /turtle1/set_pen 200 0 0 5 0

# Draw the lines by publishing to the topic /turtle1/cmd_vel
< PUBLISH TO /turtle1/cmd_vel WITH A geometry_msgs/Twist MESSAGE TO DRIVE THE TURTLE >
< EXAMPLE: >
rostopic pub -1 /turtle1/cmd_vel geometry_msgs/Twist -- '[3.0, 0.0, 0.0]' '[0.0, 0.0, 0.0]'
rostopic pub -1 /turtle1/cmd_vel geometry_msgs/Twist -- '[0.0, 0.0, 0.0]' '[0.0, 0.0, -2.62]'
rostopic pub -1 /turtle1/cmd_vel geometry_msgs/Twist -- '[3.46, 0.0, 0.0]' '[0.0, 0.0, 0.0]'
rostopic pub -1 /turtle1/cmd_vel geometry_msgs/Twist -- '[0.0, 0.0, 0.0]' '[0.0, 0.0, 2.62]'
rostopic pub -1 /turtle1/cmd_vel geometry_msgs/Twist -- '[3.0, 0.0, 0.0]' '[0.0, 0.0, 0.0]'
< REPEAT AS MANY TIMES AS NECESSARY >


# 2. Letter A

# 2a. Transport turtle1 to first point of the letter
# Turn off pen
< CALL SERVICE TO TURN OFF PEN >
rosservice call /turtle1/set_pen 200 0 0 5 1

# Transport the turtle
< CALL SERVICE TO TRANSPORT THE TURTLE >
rosservice call /turtle1/teleport_absolute '{x: 5, y: 3, theta: 1.07}'

# 2b. Draw the letter line by line

# Turn on pen
< CALL SERVICE TO TURN ON THE PEN >

# Draw the lines by publishing to the topic /turtle1/cmd_vel
< PUBLISH TO /turtle1/cmd_vel WITH A geometry_msgs/Twist MESSAGE TO DRIVE THE TURTLE >
< REPEAT AS MANY TIMES AS NECESSARY >


# 3. Letter U

# 3a. Transport turtle1 to first point of the letter
# Turn off pen
< CALL SERVICE TO TURN OFF PEN >

# Transport the turtle
< CALL SERVICE TO TRANSPORT THE TURTLE >

# 3b. Draw the letter line by line

# Turn on pen
< CALL SERVICE TO TURN ON THE PEN >

# Draw the lines by publishing to the topic /turtle1/cmd_vel
< PUBLISH TO /turtle1/cmd_vel WITH A geometry_msgs/Twist MESSAGE TO DRIVE THE TURTLE >
< REPEAT AS MANY TIMES AS NECESSARY >
