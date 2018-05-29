# AlphaZero_Quoridor
An AlphaZero implementation of game Quoridor

# 项目介绍

Chinese only now.

暂时只用中文介绍，本项目主要实现AlphaZero算法的Quoridor游戏。

## Quoridor

Quoridor中文称作步步为营，是国外非常流行的桌游，大致界面如下所示（图片来自百度图片）：

![](https://github.com/cryer/AlphaZero_Quoridor/raw/master/images/1.jpg)

游戏规则(百度百科)：

* 你可以看到本路径棋横10条、竖8条凹下的路径，20个挡板，四人不同颜色的木人。
* 游戏开始前，所有玩家（2-4）每人一个小木人，平分所有挡板。
* 游戏开始，木人放在边路，开始向对岸行驶，谁最先到达对岸者为胜利者。如果玩家为3人或4人，一人胜利之后，其他玩家还可以继续争夺第二名、第三名。你可以在棋盘的任何方向行走，每次只能一步或者放挡板。放挡板的目的是阻挡对方到达对岸，如果选择放挡板的话，木人就不能走路，如果木人走一步，就不能放挡板。需要注意的是，挡板不能把别人完全挡死，也就是说至少要留出一个出口。

* 当两子相邻的时候可以跳过对面棋子，在对面棋子的另外三个方向上都可以移动，也就是相当于走了两步

虽然该游戏可以四人玩，但是由于博弈树和AlphaZero一般处理两人博弈游戏，所以我们只考虑两人游戏的情况，两人处于对岸。每人十个挡板，挡板不可以复用，用完
就没有了，棋子先到达对岸获胜。

## 简要分析

我们知道围棋的动作空间为361，中国象棋为2068，步步为营的动作空间则为140，其中128个挡板的动作空间，64个横向和64个竖向，加上4个平时的上下左右棋子移动
和8个特殊情况的棋子移动，也就是两子相邻的情况。棋盘棋子的移动范围是9\*9。

## 棋盘状态的表示

AlphaGo Zero围棋的棋盘状态表示为48\*19\*19,其中19\*19是棋盘的大小，48是围棋的规则，这些是手工制定的，包含气等信息。而AlphaZero的围棋
棋盘状态用17\*19\*19,17是历史的落子信息，也就是说舍弃了人类的制定规则。

步步为营的棋面状态可以自行设计，只要合理就行，本个项目采用26\*9\*9,9\*9是棋面的大小，26包含竖直挡板的信息，横挡板的信息，玩家棋子位置信息，
剩余挡板数量信息和先后手信息等等，详细参考代码。

## 策略价值网络

Zero版本相比Go的版本一大改进，就是合并了策略网络和价值网络，并且用残差网络代替的普通的卷积结构，我本次也采用了残差网络，
用了5个残差块，每个残差块由2部分卷积结构组成，网络相对还是蛮深的，也只是一个初步的网络结构，后续有很大可能改进。

## GUI界面

简易的游戏界面如下：

![](https://github.com/cryer/AlphaZero_Quoridor/raw/master/images/2.png)

界面采用pygame模块构建，参考[mattdeak](https://github.com/mattdeak/QuoridorZero),只是原作者还没有完成，显示部分也有一些
BUG，我改进了一下，目前可以正常显示。

运行：
```
python game.py --player_type 2 --computer_type 1
```
就可以打开这个游戏界面，并且实现简单的人机对战，玩家先手，电脑目前是未训练的策略价值网络进行的蒙特卡洛搜索树MCTS，默认100次模拟
落子一次，这是考虑到测试时间的原因，即使只有100次，每次落子也要花费近50秒的时间，看来网络还是有些大了。最后希望800次模拟可以在90秒以内。

棋力自然是很差的，因为网络参数还没有训练，完全是初始化的值。

## 关于代码

本项目是一个长期的项目，目前还处于起步阶段，代码也是刚刚初步完成，只处于能跑通的阶段，很多地方还不完善，代码结构也没有优化，算法本身的很多参数也
调整，可以Star关注我长期的更新。

* 另外，代码很多部分我都给出了详细的中文注释，我相信可以让你更快的了解代码，让你轻松的看代码。

# Get Started

## 测试

首先需要满足以下环境：
```
python 3.X
pytorch 0.3
pygame
numpy
```
首先克隆我的代码：
```
git clone https://github.com/cryer/AlphaZero_Quoridor.git
cd AlphaZero_Quoridor/
```

对战模式有人人对战和人机对战，命令player_type指定对手类型，1是人人对战，2是人机对战，
人机对战时还可以指定对面电脑的类型，利用命令computer_type，为1时电脑类型为Alpha MCTS，也就是策略价值网络结合
蒙特卡洛树搜索的AlphaZero算法版本；为2时电脑为传统的MCTS，没有利用神经网络。

* 人人对战，运行：
```
python game.py --player_type 1
```

* 人机对战（对手为Alpha MCTS）
```
python game.py --player_type 2 --computer_type 1
```

* 人机对战（对手为传统 MCTS）
```
python game.py --player_type 2 --computer_type 2
```
后续还会加入极大极小搜索配合α-β剪枝的电脑算法，这主要是为了进行棋力的评测，因为极大极小搜索配合α-β剪枝在这个游戏上
已经可以取得非常好的效果。

## 训练

想训练，运行：

```
python train.py
```

* 注意需要根据自己的要求和电脑配置选择合适的训练参数，详细参数如下：

```
        # 训练参数
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 适应性调节学习速率
        self.temp = 1.0
        self.n_playout = 10  # 测试
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 10   # 取10 测试
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000
```
自己根据实际修改，另外由于暂时代码结构和代码书写未优化，多进程和多线程也还没有实现，所以代码的训练较慢，可以尝试，
但是最好在我加入多线程多进程之后在开始训练。



# 进度

- [x] 初步代码 :blush:
- [ ] 代码和结构优化 :worried:
- [ ] 多线程多进程 :worried:
- [ ] 算法调参 :worried:
- [ ] 训练 :worried:

## 训练进度

待续

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
