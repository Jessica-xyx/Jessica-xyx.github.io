---
layout: post
title: 最大子数组和--O(n)
---

https://leetcode.cn/problems/maximum-subarray/

用动态规划做，dp[i]表示**包含nums[i]的最大子数组和**，maxn是整个序列的最大子数组和。复杂度O(n)。

如果dp[i-1] + nums[i]的值比nums[i]小，意味着下一个连续子数组要从nums[i]开始算，也就是说nums[i]之前的数会削弱nums[i]，还不如从nums[i]开始算。

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        vector<int> dp(len, 0);
        int maxn = 0;
        dp[0] = nums[0];
        maxn = dp[0];

        for (int i = 1; i < len; i++){
            if(dp[i - 1] + nums[i] < nums[i]){
                dp[i] = nums[i];
            } 
            else{
                dp[i] = dp[i - 1] + nums[i];
            }
            maxn = max(maxn, dp[i]);
        }
        return maxn;
    }
};
```



![image-20220908215352699](/assets/img/image-20220908215352699.png)