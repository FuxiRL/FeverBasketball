At each time step, when a player needs to perform an action, 
it preceives a json data package sent from the game client which contains all the information needed to make a decision.


###  The coordinates system
<div align=center><img src=Materials/image/coordinate.png width="50%" height="50%"></div>

###  The json data package
```
{
  "end_results": [],                              // the end reselts if a state is terminated or switched, see next section for details
  "switch_result": None,                          //
  "state": "freeball",                            // ['attack', 'defense', 'freeball', 'ballclear', 'assist', 'null'], each corresponding for a game state
  "game_state":{
    "attack_remain_time": 16.2719764709473,       // the time left for current round of attack, total 20 seconds
    "game_local_time": 149.281188964844,          // the time consumed for current match 
    "game_state": "Playing",                      // current situation of the match，['AttackReady', 'BallClear', 'Playing', 'GoalIn', 'ShotclockViolation'])
                                                        //'AttackReady' occurs when a new round of attack starts
                                                        //'BallClear' occurs when the player holding ball need to get ball possession
                                                        //'Playing' occurs as the game proceeds
                                                        //'GoalIn' occurs when a player scores
                                                        //'ShotclockViolation' occurs when the offense team exceeds the time limits for attack (20s)
    "scores": [6, 8],                             // scores of both teams
    "no_camera": False
   },
  "need_action": True,                            // the indicator of a corresponding player needs to output an action
  "my_type": "18",                                // the role of a player (C:18, PF:13, PG:, SF:, SG: )
  "teams": [                                      // the information of both teams
    [                                             // the information of team0 (the home team, rendered in color)
      {                                           // the information of player0 in team0
        "position": {                             // the coordinates of player0 (see the figure at top of x-axis and z-axis, y-axis is the hight, unit: meter)
          "x": -4.42007112503052,                 // the (0,0) is in the middle of the court
          "y": 0.0,
          "z": 5.39755916595459                   
        },
        "facing": -90.0,                          // the facing angle of the player. The angle is 0 when the player is facing the bottom line, 
                                                     which gradually increases to 180 in the clockwise direction and immediately become -180 when 180 is reached.
                                                     Then it increases from -180 to 0 by continuing following the clockwise direction. 
        "fever_value": 1034.10766601563,          // the energy to unlock the ultimate skill (not available in current version)
        "fever_can_consume": True,                // whether the ultimate skill is ready
        "basket_distance": 8.02637577056885,      // the distance between the player and the basket (x=0, y=3.02, z=12.08)
        "cannot_dribble": False,                  // whether the player can dribble, (False when not possessing ball or double dribble)
        "drivein_cooltime": False,                // whether the break-through skill is cooling down (True when it is in CD)
        "in_fever_zone": False,                   // whether the player is the area to use the ultimate skill
        "dribbling": False,                       // whether the player is dribbling the ball
        "behavior": "CharacterMove",              // the current behavior of the player, [u'ThreePointFreeThrow', u'AutoIntercept', u'MikanDrill', u'Pickup', u'BlockedJumpshot', u'SkyHookShot', u'DriveinFakeBack', u'Falldown', u'Stun', u'OffenceScreen', u'Dunk', u'Stepback', u'Divingcatch', u'Rebound', u'Layup', u'Drivein', u'Dribble', u'Shoot', u'AnkleBreaker', u'Receive', u'FocusThreepointShot', u'Keep', u'SideBlock', u'CharacterMove', u'Steal', u'Postspin', u'Boxout', u'FailedShoot', u'ShotBlock', u'CrossoverBack', u'Pass', u'Defstance', u'OverdribbleAndDunk', u'FalldownAir', u'Chipout', u'Postup'])
        "shoot_rate": 12.0049648284912,           // the current shoot rate of the player
        "give_me_the_ball": False,                // whether the player is asking for the ball (can be regard as the communication signal within a team)
        "three_point_area": True,                 // whether the player is in the three-point area
        "can_steal": False,                       // whether the player can try to steal the ball (such as when doing defense)
        "can_rush_steal": False,                  // whether the player can try to do rush steal (not necessarily can success even it is True)
        "is_attacking_team": true,                // whether the player is the offense team (the team possessing ball)
        "type": "C",                              // the type of the player
        "ability_list": []                        // the ability id can be used
      },
      {                                           // the information of player1 in team0
        "position": {
          "x": -2.02229142189026,
          "y": 0.0,
          "z": 7.04619359970093
        },
        "facing": 0.0,
        "fever_value": 977.070434570313,
        "fever_can_consume": false,
        "basket_distance": 5.43603181838989,
        "cannot_dribble": false,
        "drivein_cooltime": false,
        "in_fever_zone": false,
        "dribbling": false,
        "behavior": "CharacterMove",
        "shoot_rate": 44.4201049804688,
        "give_me_the_ball": false,
        "three_point_area": false,
        "can_steal": false,
        "can_rush_steal": false,
        "is_attacking_team": true,
        "type": "C",
        "ability_list": []
      },
      {                                           // the information of player2 in team0
        "position": {
          "x": -5.03540372848511,
          "y": 0.517468333244324,
          "z": 9.79444694519043
        },
        "facing": 65.6345977783203,
        "fever_value": 978.923706054688,
        "fever_can_consume": false,
        "basket_distance": 5.54969882965088,
        "cannot_dribble": false,
        "drivein_cooltime": false,
        "in_fever_zone": false,
        "dribbling": false,
        "behavior": "Shoot",
        "shoot_rate": 66.1101303100586,
        "give_me_the_ball": false,
        "three_point_area": false,
        "can_steal": false,
        "can_rush_steal": false,
        "is_attacking_team": true,
        "type": "PG",
        "ability_list": []
      }
    ],
    [                                             // the information of team1 (the visitor team, rendered in black&white)
      {                                           // the information of player0 in team1
        "position": {
          "x": -0.0022329818457365,
          "y": 0.0,
          "z": 5.73587417602539
        },
        "facing": -57.0349769592285,
        "fever_value": 681.429748535156,
        "fever_can_consume": false,
        "basket_distance": 6.34816455841064,
        "cannot_dribble": false,
        "drivein_cooltime": false,
        "in_fever_zone": false,
        "dribbling": false,
        "behavior": "CharacterMove",
        "shoot_rate": 44.368465423584,
        "give_me_the_ball": false,
        "three_point_area": false,
        "can_steal": false,
        "can_rush_steal": false,
        "is_attacking_team": false,
        "type": "PF",
        "ability_list": []
      },
      {                                           // the information of player1 in team1
        "position": {
          "x": -1.60123836994171,
          "y": 0.0,
          "z": 3.49503350257874
        },
        "facing": -45.0,
        "fever_value": 681.683532714844,
        "fever_can_consume": false,
        "basket_distance": 8.74063873291016,
        "cannot_dribble": false,
        "drivein_cooltime": false,
        "in_fever_zone": false,
        "dribbling": false,
        "behavior": "CharacterMove",
        "shoot_rate": 5.0,
        "give_me_the_ball": false,
        "three_point_area": true,
        "can_steal": false,
        "can_rush_steal": false,
        "is_attacking_team": false,
        "type": "PG",
        "ability_list": []
      },
      {                                           // the information of player2 in team1
        "position": {
          "x": 3.06327605247498,
          "y": 0.0,
          "z": 10.5151071548462
        },
        "facing": -90.0,
        "fever_value": 681.980224609375,
        "fever_can_consume": false,
        "basket_distance": 3.42387986183167,
        "cannot_dribble": false,
        "drivein_cooltime": false,
        "in_fever_zone": false,
        "dribbling": false,
        "behavior": "CharacterMove",
        "shoot_rate": 80.8085784912109,
        "give_me_the_ball": false,
        "three_point_area": false,
        "can_steal": false,
        "can_rush_steal": false,
        "is_attacking_team": false,
        "type": "C",
        "ability_list": []
      }
    ]
  ],
  "my_index": 0,                                  // the index of the player of the team (i.e. teams[my_team][my_index])
  "my_team": 0,                                   // the index of the team
  "ball": {
    "position": {                                 // the coordinates of the ball
      "x": -4.61177444458008,
      "y": 3.14497995376587,
      "z": 10.0200901031494
    },
    "velocity": {                                 // the velocity of the ball
      "x": 0.0,                               
      "y": 0.0,
      "z": 0.0
    },
    "state": "SHOOT",                             // the state of the ball, [u'SHOOT', u'PHYSICS', u'OWNED', u'PASS'])
    "team": -1,                                   // the team index of the ball owner (-1 no team is possessing the ball)
    "owner_index": -1                             // the player index of the ball owner (-1 means nobody is holding the ball)
  },
  "is_heart_beat": false,
  "last_action": {                                // last action of the player (after the the action is performed)
    "index": 6,                                   // the index of the action
    "name": "CharacterMove_Left",                 // the name of the action
    "behavior_showed": true,                      // whether the action is successfully performed
    "state": "",                                  // the state that the player was in (sometimes it is "")
    "is_fever": false,                            // whether the player used ultimate skill
    "is_drive_in": false,                         // whether the player used breakthrough skills
    "id": [5],                                    // the id and description of the action
    "desc": "have three state, attack without ball, defense and ball on the ground"
  },
  "action_count": {                               // action spaces of different game states
    "attack": 29,
    "defense": 18,
    "freeball": 11,
    "ballclear": 21,
    "assist": 10
  }
}\


"end_results":
{
  "state": "assist",                                
                                                    //'attack': set(['two_blocked', 'time_up', 'lost', 'stealed', 'two', 'three', 'passed', 'pass_lost', 'three_blocked']), 
                                                    //'defense': set(['two_blocked', 'time_up', 'lost', 'stealed', 'two', 'three', 'three_blocked']), 
                                                    //'freeball': set(['me', 'goal_in', 'opponent']), 
                                                    //'assist': set(['time_up', 'lost', 'stealed', 'three', 'two', 'get_ball']), 
                                                    //'ballclear': set(['lost', 'passed', 'success'])}
  "result": "two",
  "last_action": {
    "index": 2,
    "name": "CharacterMove_Right",
    "behavior_showed": true,
    "state": "",
    "is_fever": false,
    "is_drive_in": false,
    "id": [
      5
    ],
    "desc": "have three state, attack without ball, defense and ball on the ground"
  }
}]


```
