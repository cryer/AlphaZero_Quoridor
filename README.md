# AlphaZero_Quoridor
An AlphaZero implementation of game Quoridor

## Quoridor


![](https://github.com/cryer/AlphaZero_Quoridor/raw/master/images/1.jpg)

## GUI


![](https://github.com/cryer/AlphaZero_Quoridor/raw/master/images/2.png)

How to run:
```
python game.py --player_type 2 --computer_type 1
```

# Get Started


```
python 3.X
pytorch 0.3
pygame
numpy
```

* Play against Human agent
```
python game.py --player_type 1
```

* Play against AlphaZero version agent
```
python game.py --player_type 2 --computer_type 1
```

* Play aginst Pure MCTS agent
```
python game.py --player_type 2 --computer_type 2
```

## Training



```
python train.py
```



# To do list

- [x] Implement a smarter winning checker function. 
- [ ] Implement a greedy Quoridor agent.
- [ ] Write an evaluation code by playing against various types of Quoridor agents.
- [ ] Parallelize MCTS simulation.

# MIT License

```

Copyright (c) 2018 kurumi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
