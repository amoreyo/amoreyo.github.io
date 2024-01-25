## 论坛的整体构思要领
1. 讨论用户的整体架构，也就是用户的话语应该出现在当前页面的哪个结构中的哪个位置
2. 论述话语的增删查改，也就是论述话语可以被他人修改并且留下版本控制


## 要点补充
1. 因为我们讨论的，来自于生活日常中的细节。相比于辩论的论点，我们讨论的点更像是一块打破水面平静的石头，所以肯定会存在高于初始的点的论述。所以我们需要增加对于论题本身审视的功能。
2. 以及，完全用reddit那套树状结构来记录对话的进行，是不合适的。reddit的论坛讨论更像是对问题的一嘴一句，也就是说，不会存在衍生出其他更具体的东西，reddit的一个讨论的点就是树状结构的叶子节点。
比方说，当我和xyh突然激烈对骂，那树状的均衡衍生就会被破坏。
所以我们需要把握住什么时候该对子话题作为一个新的讨论点，作为新的文件形式创建出来

## 框架设计
1. 以文件作为一个论点，及其父论点，及其子论点为存储对象，作为一个entity，或者说一个指针。
2. 文件中的component有{ 2 }
    1. 第一个component就是对文件代表的论点本身来进行贴合的详细的论述。
        1. 既然是详细描述，就要求用户有自律的发言，可以用纯粹的树状结构来进行，用户之间可以互相增删查改。
    2. 第二个component就是贴近闲聊的发散。
        1. 直接抄qq群聊天的形式即可，因为到处都有闲聊，所以这部分不做过多管理和约束，也不加入增删查改的功能。
    3. 文件（指针），可以存在任何位置。

example: 
