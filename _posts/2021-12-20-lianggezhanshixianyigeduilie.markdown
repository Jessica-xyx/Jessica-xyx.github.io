---
layout : post
title : 用两个栈实现一个队列
categories: [算法与刷题]
tags: 
---

<a href="https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/" >点此跳转leetcode链接<a>

 ![photo]({{site.url}}/assets/img/offer1.png)

```c++
/*

*/
class CQueue {
public:
stack<int> in_s;
        stack<int> out_s;
    CQueue() {
        
    }
    
    void appendTail(int value) {
        in_s.push(value);
    }
    
    int deleteHead() {
        if(out_s.size()==0 && in_s.size()==0)
        {
            return -1;
        }
        if(out_s.size()==0)
        {
            while(in_s.size())
            {
                int temp=in_s.top();
                out_s.push(temp);
                in_s.pop();
            }
            int temp=out_s.top();
            out_s.pop();
            return temp;
        }
        int temp=out_s.top();
        out_s.pop();
        return temp;
    }
};

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue* obj = new CQueue();
 * obj->appendTail(value);
 * int param_2 = obj->deleteHead();
 */
```

![image-20211227123748133](/assets/img/image-20211227123748133.png)