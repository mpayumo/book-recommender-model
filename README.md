# Book Recommender Model

> by: [:globe_with_meridians: Mark Payumo](https://www.linkedin.com/in/markpayumo/)

Exploratory data analysis, and matrix factorization and deep learning model development of dataset from [Book Crossing](https://www.bookcrossing.com).

## Description

![The New York Public Library](img/nyc_library.jpg "The New York Public Library by David Iliff via Wikimedia Commons")

The **goal** of this project is to demonstrate the applicability of matrix factorization and deep learning when modeling and developing a recommender system—in this case, a **book recommender system**. It utilizes the dataset that was collected by [Cai-Nicolas Ziegler](http://www2.informatik.uni-freiburg.de/~cziegler/BX/WWW-2005-Preprint.pdf) during a four-week crawl (Aug. to Sep. 2004) of the Book Crossing community.

## Data Source

> ### The dataset comes in three CSV files that separately contain the following:

<ul>
    <li>User demographic information</li>
    <li>Book ratings</li>
    <li>Book titles along with author, publisher, publication, ISBN, etc. </li>
</ul>

## Tech Stack

> ### The following were used in the development of the recommender system:

<ul>
    <li>Python</li>
    <li>JupyterLab</li>
    <li>Amazon Web Services</li>
    <li>Git</li>
<!--     <li></li> -->
</ul>

## Exploratory Data Analysis

### Rating Scale

Book Crossing has a sliding scale of 1 to 10 that users utilize to rate the books. There is an overwhelming amount of users in 2004 that gave the books a rating of 0. Overall, there is a total of <code>1,149,780</code> ratings during the specified time period.

<p align = "center"><img src = "img/distribution_book_ratings.jpg"></p>

The donut chart below shows those books that were rated "0" own the lion's share of the distribution.
<p align = "center"><img src = "img/wedge_donut.jpg"></p>

### Rated Books by Publication Year

Books published in 2002 were rated the most by 17,627 users. Those that were published between 1975 to 2002 received the most ratings.

<p align = "center"><img src = "img/timeseries.jpg"></p>

### Countries of Origin

The U.S. tops the list of users' countries of origin at 139,711. Below is just the top 20 out of 707 user-defined countries, which were extracted from the users' text addresses where many are gibberish entries that required further cleaning through text manipulation.

<p align = "center"><img src = "img/top20_countries1.jpg"></p>

---

## Modeling

## Future Direction