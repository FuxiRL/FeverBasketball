There are several args for FeverBasketball client (i.e. building.exe) configuration:
* **-n**: game mode configuration (1v1, 2v2, 3v3), choose from [1,2,3]
* **-i**: ip address of the server (i.e. run_server.py), default is 127.0.0.1
* **-o**: port of the server, default is 6666
* **-x**: team member configuration, six digits in total, 
where the first three is for team0 (the home team, rendered in color) and the last three is for team1 (the visitors, rendered in black&white).
0-PG, 1-SG, 2-SF, 3-PF, 4-C. Notice that the latter digits will be ignored when the number of the team is less than three.
* **-p**: for human play, choose from [0,1,2] to control corresponding players in team0. The default number is -1, which means no human control. 
(Notice that the controlling of the firtst player (-p 0) in built-in AI pack is currently not available.
You can try '-p 1' or '-p 2' to control the first or the second player in the host team together with your trained agents to play against the built-in AI, 
and we will fix this soon.)

Examples:

`building.exe -n 1 -o 5000 -x 124034 -p -1`  
Meaning: Starting a 1v1 FeverBasketball clinet to connect local server 127.0.0.1 on port 5000 without human control. 
The player for team0 is SG and for team1 is PG.  

`building.exe -n 3 -i 192.168.1.154 -o 6666 -x 124124 -p -1`  
Meaning: Starting a 3v3 FeverBasketball clinet to connect server 192.168.1.154 on port 6666 without human control. 
The players for both teams are the same, namely [SG, SF, C].  

`building.exe -n 3 -i 192.168.1.154 -o 6666 -x 124124 -p 2`  
Meaning: Starting a 3v3 FeverBasketball clinet to connect server 192.168.1.154 on port 6666 with human controlling the third player of team0. 
The players for both teams are the same, namely [SG, SF, C].  