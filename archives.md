---
layout: page
title: Archive
permalink: /archives/
---

<div class="archive">
{% assign posts_by_year = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}
{% for year in posts_by_year %}
  <section class="archive-year">
    <h2 class="archive-year-title">{{ year.name }} <span class="archive-count">({{ year.items.size }})</span></h2>
    {% assign by_month = year.items | group_by_exp: "post", "post.date | date: '%m'" | sort: "name" | reverse %}
    {% for month in by_month %}
      {% assign first_post = month.items | first %}
      <h3 class="archive-month-title">{{ first_post.date | date: "%B" }} <span class="archive-count">({{ month.items.size }})</span></h3>
      <ul class="archive-list">
        {% assign month_posts = month.items | sort: "date" | reverse %}
        {% for post in month_posts %}
          <li class="archive-item">
            <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>
            <span class="archive-date">{{ post.date | date: "%B %e, %Y" }}</span>
          </li>
        {% endfor %}
      </ul>
    {% endfor %}
  </section>
{% endfor %}
</div>
