---
layout: default
category: about
title: 云迹
description: 见到你很高兴
---

<section class="about-me inner">
  <h1>
    <span class="hello">
      <em>关于我</em>
    </span>
  </h1>
</section>

<!-- <section class="about-me inner">
  <p>
     <strong> 研究方向 </strong>
     <ul class="show-list">
     	<li>multi-agent RL</li>
     	<li>cloud computing</li>
     </ul>
  <p>
</section> -->

<section class="about-me inner">
<h2>
    <span class="hello">
      <em>学历</em>
    </span>
  </h2>
  <ul class="show-list">
     	<li>2018-2022 北京交通大学 本科 计算机科学与技术专业</li>
     	<li>2022-2025 北京交通大学 硕士研究生 计算机科学与技术专业 云计算方向</li>
     </ul>

  <h2>
    <span class="hello">
      <em>项目&实习经历</em>
    </span>
  </h2>
  <ul class="show-list">
     	<li>2020年2月至-2021年4月 深度注意力引导的实例级显著性目标检测---->校级大创</li>
     	<li>2021年8月至9月 阿里巴巴本地生活 助理测试工程师</li>
     	<li>2022年5月至8月 字节跳动商业化技术中台 测试开发实习生-客户端方向</li>
     </ul>



<h2>
    <span class="hello">
      <em>获奖</em>
    </span>
  </h2>
  <ul class="show-list">
     	<li>校三好学生（2018-2019）</li>
     	<li>二等学习优秀奖学金（2018-2019）</li>
     	<li>2020年全国大学生数学建模竞赛北京市二等奖</li>
        <li>2021年美国大学生数学建模竞赛H奖</li>
        <li>2021年北京交通大学数学建模竞赛三等奖</li>
     </ul>


<h2>
    <span class="hello">
      <em>语言</em>
    </span>
  </h2>
  <ul class="show-list">
     	<li>CET-6 597分</li>
     </ul>
</section>





<section class="inner">
  <div class="post-recents-in-index">
    <strong>Recent blog posts</strong>
  </div>
  <blockquote>
    <ul class="blog-list">
      {% for post in site.posts limit:5 %}
      <li class="post-list-in-index">
      <small class="post-list-date-in-index">{{ post.date | date_to_string }}</small>
      <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
      </li>
      {% endfor %}
    </ul>
  </blockquote>
  <div class="spacer"></div>
</section>

<section class="inner">
  <p>
    <ul class="show-list">
      <li>Email: <a href="https://mail.163.com/">xieyixuancs@163.com</a></li>
      <li>Github: <a href="https://github.com/Jessica-xyx">github.com/Jessica-xyx</a></li>
    </ul>
  </p>
</section>

<!-- <section class="inner">
  <p>
    If you are interested in me, you can access to my <a href="/nijiazhi_resume.pdf"><strong>Resume</strong></a> for more details.
    <ul class="show-list">
      <li>Email: <a href="https://mail.qq.com/">954142793@qq.com</a></li>
      <li>Google Scholar: <a href="https://scholar.google.com/citations?user=hHi46EcAAAAJ">Jiazhi Ni</a></li>
      <li>Kaggle: <a href="https://www.Kaggle.com/nijiazhi">Andy</a></li>
      <li>Github: <a href="https://github.com/Jessica-xyx">github.com/Jessica-xyx</a></li>
      <li>知乎：<a href="https://www.zhihu.com/people/andy-3-36/activities">写代码的段子手</a></li>
      <li>码云Git：<a href="https://gitee.com/nijiazhi/events">gitee.com/nijiazhi</a></li>
      <li>开源中国：<a href="https://my.oschina.net/njz">njz_andy</a></li>
    </ul>
  </p>
</section> -->