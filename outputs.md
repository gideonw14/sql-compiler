Key:
```
Tree
\
 |------Branch
        \
                |------Right branch
               /
        |------Node
               \
                |-------Left Branch
                

PROJECT [Attributes]
\
 |------SELECT [Conditions]
        \
                |------[Attributes] (Project before join)
                       \
                        |------[[Condtions]] (Select before join)
                               \
                                |------Relation (to be joined)
               /
        |------|><| (join with conditions) or X (cross product)
               \
                |-------Left Branch
```


####A
Relational Algebra

`PROJECT [S.SID, S.SNAME, S.RATING, S.AGE] (SELECT [S.RATING > 7] (SAILORS AS S))`

Query Tree (optimized tree the same)
```
PROJECT [S.SID, S.SNAME, S.RATING, S.AGE]
\
 |------['S.RATING > 7']
        \
         |------SAILORS AS S
```

####B
`Exception: Attribute COLOR is not in the attributes for relation SAILORS`
####C
Relational Algebra

`PROJECT [B.COLOR] (SELECT [S.SID = R.SID AND R.BID = B.BID AND S.SNAME = LUBBER] (SAILORS AS S X [RESERVES AS R X [BOATS AS B]]))`

Initial Query Tree

```
PROJECT [B.COLOR]
\
 |------SELECT [S.SID = R.SID AND R.BID = B.BID AND S.SNAME = LUBBER]
        \
                 |------['SAILORS', 'S']
                /
         |------X
                \
                         |------['BOATS', 'B']
                        /
                 |------X
                        \
                         |------['RESERVES', 'R']
```

Optimized Query Tree
```
PROJECT [B.COLOR]
\
         |------[S.SID]
                \
                 |------[['S.SNAME = LUBBER']]
                        \
                         |------SAILORS AS S
        /
 |------|><| S.SID = R.SID
        \
                 |------[B.BID, B.COLOR]
                        \
                         |------BOATS AS B
                /
         |------|><| R.BID = B.BID
                \
                 |------[R.SID, R.BID]
                        \
                         |------RESERVES AS R
```
####D
Relational Algebra

`PROJECT [SNAME] (SELECT [SAILORS.SID = RESERVES.SID AND RESERVES.BID = BOATS.BID AND BOATS.COLOR = RED OR BOATS.COLOR = GREEN] (SAILORS X [BOATS X [RESERVES]]))`

Initial Query Tree

```
PROJECT [SNAME]
\
 |------SELECT [SAILORS.SID = RESERVES.SID AND RESERVES.BID = BOATS.BID AND BOATS.COLOR = RED OR BOATS.COLOR = GREEN]
        \
                 |------['SAILORS']
                /
         |------X
                \
                         |------['RESERVES']
                        /
                 |------X
                        \
                         |------['BOATS']
```

Optimized Query Tree
```
PROJECT [SAILORS.SNAME]
\
         |------[SAILORS.SID, SAILORS.SNAME]
                \
                 |------SAILORS
        /
 |------|><| SAILORS.SID = RESERVES.SID
        \
                 |------RESERVES
                /
         |------|><| RESERVES.BID = BOATS.BID
                \
                 |------[BOATS.BID]
                        \
                         |------[['BOATS.COLOR = RED OR', 'BOATS.COLOR = GREEN']]
                                \
                                 |------BOATS

```

####E
`Exception: Attribute RATING is not in the attributes for relation RESERVES`
####F
Relational Algebra

`PROJECT [SNAME] (SELECT [SAILORS.SID = RESERVES.SID AND RESERVES.BID = BOATS.BID AND BOATS.COLOR = RED AND BOATS.COLOR = GREEN] (SAILORS X [BOATS X [RESERVES]]))`

Initial Query Tree

```
PROJECT [SNAME]
\
 |------SELECT [SAILORS.SID = RESERVES.SID AND RESERVES.BID = BOATS.BID AND BOATS.COLOR = RED AND BOATS.COLOR = GREEN]
        \
                 |------['SAILORS']
                /
         |------X
                \
                         |------['RESERVES']
                        /
                 |------X
                        \
                         |------['BOATS']
```

Optimized Query Tree
```
PROJECT [SAILORS.SNAME]
\
         |------[SAILORS.SID, SAILORS.SNAME]
                \
                 |------SAILORS
        /
 |------|><| SAILORS.SID = RESERVES.SID
        \
                 |------RESERVES
                /
         |------|><| RESERVES.BID = BOATS.BID
                \
                 |------[BOATS.BID]
                        \
                         |------[['BOATS.COLOR = RED AND', 'BOATS.COLOR = GREEN']]
                                \
                                 |------BOATS
```
####G
Relational Algebra

`PROJECT [S.SID] (SELECT [S.SID = R.SID AND R.BID = B.BID AND B.COLOR = RED AND NOT S2.SID = R2.SID AND R2.BID = B2.BID AND B2.COLOR = GREEN] (SAILORS AS S X [RESERVES AS R X [BOATS
 AS B X [SAILORS AS S2 X [RESERVES AS R2 X [BOATS AS B2]]]]]))`

Initial Query Tree

```
PROJECT [S.SID]
\
 |------SELECT [S.SID = R.SID AND R.BID = B.BID AND B.COLOR = RED AND NOT S2.SID = R2.SID AND R2.BID = B2.BID AND B2.COLOR = GREEN]
        \
                 |------['SAILORS', 'S']
                /
         |------X
                \
                         |------['RESERVES', 'R']
                        /
                 |------X
                        \
                                 |------['BOATS', 'B']
                                /
                         |------X
                                \
                                         |------['SAILORS', 'S2']
                                        /
                                 |------X
                                        \
                                                 |------['BOATS', 'B2']
                                                /
                                         |------X
                                                \
                                                 |------['RESERVES', 'R2']
```

Optimized Query Tree
```
PROJECT [S.SID]
\
         |------[S.SID, S.SID]
                \
                 |------SAILORS AS S
        /
 |------|><| S.SID = R.SID
        \
                 |------[R.SID, R.BID]
                        \
                         |------RESERVES AS R
                /
         |------|><| R.BID = B.BID
                \
                         |------[B.BID]
                                \
                                 |------[['B.COLOR = RED']]
                                        \
                                         |------BOATS AS B
                        /
                 |------X
                        \
                                 |------[S2.SID]
                                        \
                                         |------SAILORS AS S2
                                /
                         |------|><| S2.SID = R2.SID
                                \
                                         |------[B2.BID]
                                                \
                                                 |------[['B2.COLOR = GREEN']]
                                                        \
                                                         |------BOATS AS B2
                                        /
                                 |------|><| R2.BID = B2.BID
                                        \
                                         |------[R2.SID, R2.BID]
                                                \
                                                 |------RESERVES AS R2
```
####H
Relational Algebra

`PROJECT [S.SNAME] (SELECT [S.SID = R.SID AND R.BID = 103] (SAILORS AS S X [RESERVES AS R]))`

Initial Query Tree

```
PROJECT [S.SNAME]
\
 |------SELECT [S.SID = R.SID AND R.BID = 103]
        \
                 |------['RESERVES', 'R']
                /
         |------X
                \
                 |------['SAILORS', 'S']
```

Optimized Query Tree
```
PROJECT [S.SNAME]
\
         |------[R.SID]
                \
                 |------[['R.BID = 103']]
                        \
                         |------RESERVES AS R
        /
 |------|><| S.SID = R.SID
        \
         |------[S.SNAME, S.SID]
                \
                 |------SAILORS AS S
```
####I
`Exception: Relation RESERVE not in the database.`
####J
Relational Algebra

`PROJECT [S.SNAME] (SELECT [R.BID = B.BID AND R.SID = S.SID] (SAILORS AS S X [RESERVES AS R X [BOATS AS B]]))`

Initial Query Tree

```
PROJECT [S.SNAME]
\
 |------SELECT [R.BID = B.BID AND R.SID = S.SID]
 \
          |------['SAILORS', 'S']
         /
  |------X
         \
                  |------['RESERVES', 'R']
                 /
          |------X
                 \
                  |------['BOATS', 'B']
```

Optimized Query Tree
```
PROJECT [S.SNAME]
\
         |------[S.SNAME, S.SID]
                \
                 |------SAILORS AS S
        /
 |------|><| R.SID = S.SID
        \
                 |------[R.BID, R.SID]
                        \
                         |------RESERVES AS R
                /
         |------|><| R.BID = B.BID
                \
                 |------[B.BID]
                        \
                         |------BOATS AS B
```
####K
Relational Algebra

`PROJECT [S.SNAME] (SELECT [S.AGE > MAX(S2.AGE) AND S2.RATING = 10] (SAILORS AS S X [SAILORS AS S2]))`

Initial Query Tree

```
PROJECT [S.SNAME]
\
 |------SELECT [S.AGE > MAX(S2.AGE) AND S2.RATING = 10]
        \
                 |------['SAILORS', 'S2']
                /
         |------X
                \
                 |------['SAILORS', 'S']
```

Optimized Query Tree
```
PROJECT [S.SNAME]
\
         |------[['S2.RATING = 10']]
                \
                 |------SAILORS AS S2
        /
 |------X
        \
         |------[S.SNAME]
                \
                 |------[['S.AGE > MAX(S2.AGE)']]
                        \
                         |------SAILORS AS S
```
####L
Relational Algebra

`GROUP BY [B.BID] (PROJECT [B.BID, COUNT(*) AS RESERVATIONCOUNT] (SELECT [R.BID = B.BID AND B.COLOR = RED] (BOATS AS B X [RESERVES AS R])))`

Initial Query Tree

```
GROUP BY [B.BID]
\
 |------PROJECT [B.BID, COUNT(*) AS RESERVATIONCOUNT]
        \
         |------SELECT [R.BID = B.BID AND B.COLOR = RED]
                \
                         |------['RESERVES', 'R']
                        /
                 |------X
                        \
                         |------['BOATS', 'B']
```

Optimized Query Tree
```
GROUP BY [B.BID]
\
 |------PROJECT [B.BID, COUNT(*) AS RESERVATIONCOUNT]
        \
                 |------[R.BID]
                        \
                         |------RESERVES AS R
                /
         |------|><| R.BID = B.BID
                \
                 |------[B.BID, B.BID]
                        \
                         |------[['B.COLOR = RED']]
                                \
                                 |------BOATS AS B
```
####M
Relational Algebra

`HAVING [B.COLOR = RED] (GROUP BY [B.BID] (PROJECT [B.BID, COUNT(*) AS RESERVATIONCOUNT] (SELECT [R.BID = B.BID AND B.COLOR = RED] (BOATS AS B X [RESERVES AS R]))))`

Initial Query Tree

```
HAVING [B.COLOR = RED]
\
 |------GROUP BY [B.BID]
        \
         |------PROJECT [B.BID, COUNT(*) AS RESERVATIONCOUNT]
                \
                 |------SELECT [R.BID = B.BID AND B.COLOR = RED]
                        \
                                 |------['RESERVES', 'R']
                                /
                         |------X
                                \
                                 |------['BOATS', 'B']
```

Optimized Query Tree
```
HAVING [B.COLOR = RED]
\
 |------GROUP BY [B.BID]
        \
         |------PROJECT [B.BID, COUNT(*) AS RESERVATIONCOUNT]
                \
                         |------[R.BID]
                                \
                                 |------RESERVES AS R
                        /
                 |------|><| R.BID = B.BID
                        \
                         |------[B.BID, B.BID]
                                \
                                 |------[['B.COLOR = RED']]
                                        \
                                         |------BOATS AS B
```
####N
`Exception: Relation or alias SAILOR is not used in this query`
####O
`Exception: Invalid syntax near or at "AVE ("`