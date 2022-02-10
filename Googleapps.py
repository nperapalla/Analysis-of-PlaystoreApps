import pandas as pd
import numpy as np
import re
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import plotly.express as px
from wordcloud import WordCloud,STOPWORDS
import warnings
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

path = "C:/Users/nihar/Desktop/Datascience project/Analysis_of_PlaystoreApps/Datasets/"
appfile = pd.read_csv(path+'app.csv',encoding='UTF-8')
comments=pd.read_csv(path+'comment.csv',encoding='UTF-8')
df=pd.DataFrame(appfile)#,columns=['id','app_name','genre','rating','reviews','price','rate5','rate4','rate3','rate2','rate1','updated','size','installs','current_version','required_android','content_rating','in-app price','company'])
df.dropna(axis=0,how='all',inplace=True,subset=['app_name','genre','rating','reviews','cost_label','rate_5_pc','rate_4_pc','rate_3_pc','rate_2_pc','rate_1_pc','updated','size','installs','current_version','requires_android','content_rating','in_app_products','offered_by'])
df.drop(['in_app_products'],axis=1,inplace=True)
df.dropna(axis=0,how='all',inplace=True,subset=['rating'])
print(len(df))
dfc1=pd.DataFrame(comments)

#inserting category column
df['category']=df['genre']
df['category'].replace(['Education','Educational'],'Education',inplace=True)    #'Casual''Simulation','Strategy','Trivia'
df['category'].replace(['Music','Music & Audio'],'Music',inplace=True)
df['category'].replace(['Action','Adventure','Arcade','Board','Card','Casino','Puzzle','Racing','Role Playing','Word'],'Game',inplace=True)

# reviews column
df['reviews'] =  [re.sub(r'[^\d]','', str(x)) for x in df['reviews']]
df.reviews=pd.to_numeric(df.reviews).astype(np.int64)

#removing unnecesary values in costlabel & converting to USD
df['cost_label'] =  [re.sub(r'[,₫]','', str(x)) for x in df['cost_label']]
df['cost_label']=  df['cost_label'].apply(lambda x: re.findall(r'[0-9]+', str(x)))
df['cost_label']=  df['cost_label'].apply(lambda x: 0 if x==[] else x[-1])
df.cost_label=df.cost_label.astype(np.int64)
df['price in USD']=df.cost_label *0.000043
np.around(df['price in USD'],decimals=2)

#Removing + from the installs column & converting to integer
df['installs'] =  [re.sub(r'[^\d]','', str(x)) for x in df['installs']]
df.installs=pd.to_numeric(df.installs).astype(np.int64)

#Adding a column type based on Price column
df['type']=df['price in USD'].apply(lambda x: 'free' if x==0 else 'paid')

#converting all sizes to Kb
df['size'] = df['size'].apply(lambda x: str(x)).apply(lambda x: re.sub(r'[,]','', str(x)))
def convert_kb(x):
    if x.endswith('M'):
        y=x[:-1]
        return float(y)*1000
    elif x.endswith('k'):
        y=x[:-1]
        return float(y)
    else:return 0
df['size'] = df['size'].apply(lambda x: convert_kb(x))

#Convert updated column to datetime
df['updated']= pd.to_datetime(df['updated'])
print(type(df['updated'].iat[0]))
print(df[:10])
df['year']=df['updated'].dt.year
df['month']=df['updated'].dt.month

print(df.info())
print(df.isnull().sum())

warnings.filterwarnings("ignore", category=RuntimeWarning)


#No.of apps in each
bins=[1,1.5,2,2.5,3,3.5,4,4.5,5]
c1 = pd.cut(df['rating'], bins)
c2=pd.value_counts(c1)
print("Total No. of apps in each category :\n ",c2)
fig=plt.figure()
plt.hist(df['rating'],bins=8,color='purple')
plt.title("No. of apps in each rating",fontsize=12)
plt.axvline(df['rating'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.xlabel("Rating",fontsize=12)
plt.ylabel("No. of apps",fontsize=12)

#count based on installs
fig=plt.figure()
ax = sns.countplot(x="installs", data=df)
plt.xticks(rotation=90)
plt.title(label="Count of apps based on installs")
installsmean=df.groupby('installs')['rating'].mean()
print(installsmean)

#Maximum & min price of paid apps
pd.set_option('display.max_rows',None,'display.max_columns',None)
paid=df[df['price in USD']!=0]
print("Application with the highest price is ",paid['price in USD'].max())
print("Application with low price",paid['price in USD'].min())


#genre wise mean rating & count of the apps genre wise
app_count_genre1=df.genre.value_counts()                     #gives the no.of apps genre wise
x=app_count_genre1.index
y=app_count_genre1.values
#Mean rating for each genre
genremean=df.groupby('genre')['rating'].mean().reset_index()        #mean rating of apps genre wise
print("genre wise mean rating is :\n ",genremean)

#category wise mean rating & count of apps category wise
app_count_genre=df.category.value_counts().reset_index(name='count').rename(columns={'index':'category'})     #gives the no.of apps category wise
print(app_count_genre)
x1=app_count_genre['category']
y1=app_count_genre['count']
#Mean rating for each category
categorymean=df.groupby('category')['rating'].mean().reset_index().sort_values(by='rating',ascending=False).reset_index(drop=True)
print("category wise mean rating is :\n ",categorymean.head(15))
x2=categorymean['category']
y2=categorymean['rating']

#Plot showing no of.apps categry & genre wise
fig, (ax1,ax2)=plt.subplots(1,2)
fig.subplots_adjust(wspace=0.5)
ax1.barh(x,y)
ax2.barh(x1,y1)
ax1.title.set_text("Number of applications genre wise")
ax2.title.set_text("Number of applications category wise")
ax1.set_xlabel('count', fontsize=10)
ax2.set_xlabel('count', fontsize=10)
ax1.set_ylabel('Genre', fontsize=9)
ax2.set_ylabel('category', fontsize=9)

#Plots showing the top categories based on No. of apps & mean rating
fig, (ax1,ax2)=plt.subplots(1,2)
fig.subplots_adjust(wspace=0.5)
ax1.barh(x1[:15],y1[:15],color=['black', 'red', 'green', 'blue', 'cyan'])
ax2.barh(x2[:15],y2[:15],color=['black', 'red', 'green', 'blue', 'cyan'])
ax1.title.set_text("Top categories based on no. of application")
ax2.title.set_text("Top categories based on mean rating")
ax1.set_xlabel('count', fontsize=11)
ax2.set_xlabel('rating', fontsize=11)
ax1.set_ylabel('category', fontsize=11)
ax2.set_ylabel('category', fontsize=11)

#Trends of no.of apps updated in each top category obtained from above year & month wise
df_trends=df.pivot_table(index='year',columns='category',values='id',aggfunc='count').fillna(0)
df_trendsm=df.pivot_table(index='month',columns='category',values='id',aggfunc='count').fillna(0)
subset=df_trends[['Casual','Books & Reference','Education','Game','Lifestyle','Health & Fitness','Music']]
subset1=df_trendsm[['Casual','Books & Reference','Education','Game','Lifestyle','Health & Fitness','Music']]
fig, (ax1,ax2)=plt.subplots(1,2)
fig.subplots_adjust(wspace=0.25)
subset.plot(figsize=(10,8),kind='line',marker='o',ax=ax1,grid=True,ylabel='count',title = "Trends of updated apps each year in top categories",xticks=range(2010,2021,1))
subset1.plot(figsize=(10,8),kind='line',marker='o',ax=ax2,grid=True,ylabel='count',title = "Trends of updated apps each month in top categories",xticks=range(1,13,1))



#paid apps with highest price & lowest price
dfd=df[['app_name','rating','installs','genre','price in USD']]
pd.set_option('display.max_rows',None,'display.max_columns',None)
dfp=dfd[dfd['price in USD']!=0]
dfps=dfp.sort_values(['price in USD'],ascending=True).reset_index(drop=True)
print("App with the highest price is:\n",dfps.head(1))
print("App with the lowest price is :\n",dfps.tail(1))


#Preferred category by devlopers
developers_categorywise=df.groupby('category')['offered_by'].count().reset_index().sort_values(by='offered_by',ascending=False).reset_index(drop=True)
sum=developers_categorywise['offered_by'][10:37].sum()
print(sum)
developers_categorywise.drop(developers_categorywise.index[10:37],axis=0,inplace=True)
developers_categorywise.loc[len(developers_categorywise.index)]=['other',sum]
print(developers_categorywise)
fig=plt.figure()
sns.barplot(data=developers_categorywise,x='category',y='offered_by')
plt.xlabel("Category",fontsize=12)
plt.ylabel("Developers",fontsize=12)
plt.xticks(rotation=20)

#Which age group allowed apps are given more ratings & reviews.
rating_agegroup=df.groupby('content_rating')['rating'].count()#.reset_index().rename(columns={'rating':'rating_count'})         #excluding empty ones to include use id instead of rating
print(rating_agegroup)
reviews_agegroup=df.groupby('content_rating')['reviews'].sum()#.reset_index().rename(columns={'reviews':'reviews_sum'})
print(reviews_agegroup)

fig, (ax1,ax2)=plt.subplots(1,2)
fig.subplots_adjust(wspace=0.10)
rating_agegroup.plot(kind='pie',ax=ax1)
ax1.yaxis.label.set_visible(False)
ax1.set_title("Count of rating of apps in each age group",fontsize=13)
reviews_agegroup.plot(kind='pie',ax=ax2)
ax2.yaxis.label.set_visible(False)
ax2.set_title("Total No. of reviews given in each age group",fontsize=13)

#No. of paid apps & free apps & #trends of free & paid apps year wise
app_count_cost=df.groupby('type')['id'].count()
print("\n No. of free & paid applications are " ,app_count_cost)
fig , (ax1,ax2)=plt.subplots(1,2)
fig.subplots_adjust(wspace=0.5)
plt.subplot(121)
plt.pie(app_count_cost.tolist(),autopct='%1.0f%%',colors=['skyblue','orange'])#plt.pie(app_count_cost.tolist(),labels=['free','paid'],autopct='%1.0f%%')
plt.legend(labels=['free','paid'],title='Type', bbox_to_anchor=(1,1))
plt.title("Percentage of free & paid apps",fontsize=12)
plt.subplot(122)
sns.countplot(df['year'],hue=df['type'],palette=['#55CEFF','#FFA500'])
plt.title("Number of paid/free applications year wise")
ax2.tick_params(axis='x',labelrotation=50)

#year wise avg rating
c=df.groupby('year').rating.mean()
year_rating=c.plot(x='year',y='rating',kind='bar',color='purple')

df_mean=df.pivot_table(index='year',columns='type',values='rating',aggfunc='mean').fillna(0)
df_mean_genre=df.pivot_table(index='category',columns='type',values='rating',aggfunc='mean').fillna(0)

fig, (ax1,ax2)=plt.subplots(1,2)
df_mean_genre.loc[:,['free','paid']].plot.barh(ax=ax1)
df_mean.loc[:,['free','paid']].plot.bar(stacked=False,ax=ax2)
ax1.title.set_text("mean rating of paid/free applications category wise")
ax2.title.set_text("mean rating of paid/free applications year wise")
ax2.set_ylabel('Mean rating')

#top genres in each content rating category
b = df.groupby(['content_rating', 'category']).agg({'rating': 'mean'})
contentrating= b.groupby(level=0, group_keys= False).apply(lambda x: x.sort_values(by='rating',ascending=False).head(1))
print(contentrating)
#my_colors=['r','g','b','k','y','m','c']
top_genre=contentrating.plot(kind='barh',title='Top genre in each contentrating based on rating of apps',color=['black', 'red', 'green', 'blue', 'cyan'],legend=None)
plt.xlabel("rating")
plt.xticks(rotation=45)

#top apps based on no.of installs
df2=df[['rating','installs','app_name','genre','offered_by']]
top_apps_install=df2.sort_values('installs',ascending=False).reset_index(drop=True).head(15)
pd.set_option("display.max_rows",None,"display.max_columns",None)
print(top_apps_install)

#top apps based on rating
df3=df[['rating','app_name','genre','offered_by']]
top_apps_rating=df3.sort_values('rating',ascending=False).reset_index(drop=True).head(15)
print(top_apps_rating)

fig,(ax1,ax2)=plt.subplots(1,2)
fig.subplots_adjust(wspace=0.5)
top_apps_install[["installs","app_name"]].plot(x='app_name',ylabel='installs',ax=ax1,title='Top apps based on installs',kind="bar",color=(0.8, 0.4, 0.6, 0.8),legend=None)
#plt.xticks(rotation=50,ha='right',fontsize=8)
#ax1.tick_params('x',rotation=90,labelsize=8)
gc.collect()
top_apps_rating[["rating","app_name"]].plot(x='app_name',ylabel='rating',ax=ax2,title='Top apps based on rating',kind="bar",legend=None,color=(0.8, 0.4, 0.6, 0.8))
#ax2.tick_params('x',rotation=90,labelsize=8)
gc.collect()

#top apps based on rating by giving cutoof to installations
df4=df[['rating','installs','id','app_name','genre','category','offered_by','content_rating']]
df5=df4[(df4['installs']>=1000) & (df4['installs']<=10000000)]      #df5=df4.query("installs>=1000 & installs<=10000000")
top_apps_rateinstals=df5.sort_values('rating',ascending=False).reset_index(drop=True)
top_apps_rateinstals1=top_apps_rateinstals.head(15)

#Apps which are highly rated but not popular
df6=df4[(df4['installs']<=500) &(df4['rating']>=4.5)]
Notpopular=df6.sort_values('rating',ascending=False).reset_index(drop=True)[2:16]
my_colors1=['r','g','b','k','y','m','c']
top_apps_rateinstals1[["rating","app_name"]].plot(x='app_name',ylabel='installs',title='Top apps based on installs& rating',kind="bar",color=my_colors1,legend=None)
gc.collect()
my_colors2=['r','g','b','k','y','m','c']
Notpopular[["rating","app_name"]].plot(x='app_name',ylabel='installs',title='Highly rated but not popular',kind="bar",color=my_colors2,legend=None)
gc.collect()

#Which company is offering most top 50 apps & what genre dominates them.
top_ri=top_apps_rateinstals.head(50).pivot_table(index=['offered_by'],aggfunc='size')
topappri=top_ri.reset_index()
topappri.columns=['offered_by','count']
indexname=topappri[topappri['count']==1].index
topappri.drop(indexname,inplace=True)
print(topappri.reset_index())

top_app_genre=top_apps_rateinstals.head(50).pivot_table(index=['category'],aggfunc='size')
top_app_genre1=top_app_genre.reset_index()
top_app_genre1.columns=['category','count']
print(top_app_genre1.sort_values('count',ascending=False).reset_index().head(5).drop(['index'],axis=1))

top_app_cat=top_apps_rateinstals.head(50).pivot_table(index=['content_rating'],aggfunc='size')
top_app_cat1=top_app_cat.reset_index()
top_app_cat1.columns=['content_rating','count']
print(top_app_cat1.sort_values('count',ascending=False).reset_index().head(5).drop(['index'],axis=1))

#Impact of rating on other attributes
fig, axes=plt.subplots(2,2, figsize=(10,20))
fig.suptitle("Rating vs other attributes", fontsize=13)
fig.subplots_adjust(wspace=0.20)
axes=axes.flatten()
sns.regplot(x='rating',y='installs',data=df,ax=axes[0])
sns.regplot(x='rating',y='reviews',data=df,ax=axes[1])
sns.regplot(x='rating',y='size',data=df,ax=axes[2])
sns.violinplot(y='rating',x='content_rating',data=df,ax=axes[3])
axes[3].set_xticklabels(axes[3].get_xticklabels(),rotation=20)

fig=plt.figure()
plt.title("Rating vs Category",fontsize=13)
sns.boxplot(data=df,x='rating',y='category',palette='rainbow')

#Impact of reviews on other attributes
fig, axes=plt.subplots(2,2, figsize=(10,20))
fig.suptitle("reviews vs other attributes", fontsize=13)
fig.subplots_adjust(wspace=0.20)
axes=axes.flatten()
sns.regplot(x='reviews',y='installs',data=df,ax=axes[0])
sns.regplot(x='reviews',y='price in USD',data=df,ax=axes[1])
sns.regplot(x='reviews',y='size',data=df,ax=axes[2])
sns.regplot(x='reviews',y='rate_5_pc',data=df,ax=axes[3])
#sns.violinplot(y='reviews',x='content_rating',data=df,ax=axes[3])
axes[3].set_xticklabels(axes[3].get_xticklabels(),rotation=20)

#Which star rated comments are more helpful for the users
helpful_comments=dfc1.groupby('stars')['helpfuls'].sum().reset_index()
plt.figure()
sns.barplot(x="stars",y="helpfuls",data=helpful_comments)
plt.xlabel(xlabel="stars",fontsize=12)
plt.ylabel(ylabel="upvoted",fontsize=12)

#Wordcloud generation
stopwords=set(STOPWORDS)
stopwords.update(["game","please","seem","app","go","will","got","let","see","able","star","say","made","much","but","still","now","says","put","kid"])
stopwords1=set(STOPWORDS)
stopwords1.update(["game","please","seem","app","go","will","got","let","see","able","star","say","made","much","but","still","good","now","says","put","kid"])

path1="C:/Users/nihar/Desktop/Datascience project/Analysis_of_PlaystoreApps/"
star_mask = np.array(Image.open(path1 + 'star.jpg'))
text = " ".join(review for review in dfc1.content.astype(str))
#print ("There are {} words in the combination of all review.".format(len(text)))
plt.figure()
wc = WordCloud(max_font_size=50, max_words=100, background_color="white",stopwords=stopwords,mask=star_mask).generate(text)
plt.imshow(wc, interpolation="bilinear")
plt.title("Wordcloud for all the reviews")
plt.axis("off")

dfcs1=dfc1[dfc1.stars==5]
text1 = " ".join(review for review in dfcs1.content.astype(str))
wc1 = WordCloud(max_font_size=50, max_words=100, background_color="white",stopwords=stopwords).generate(text1)

dfcs5=dfc1[dfc1.stars==1]
text5 = " ".join(review for review in dfcs5.content.astype(str))
wc2 = WordCloud(max_font_size=50, max_words=100, background_color="white",stopwords=stopwords1).generate(text5)

fig,(ax1,ax2)=plt.subplots(1,2)
ax1.imshow(wc1)
ax1.title.set_text("Word cloud for 5 star ratings")
ax1.axis("off")
ax2.imshow(wc2)
ax2.title.set_text("Word cloud for 1 star ratings")
ax2.axis("off")

#Top developers wich are producing apps in more than 1 category
developers_categorywise=df.groupby(['category','offered_by'])['app_name'].count().reset_index()
pd.set_option('display.max_rows',None,'display.max_columns',None)
dev_df=developers_categorywise.sort_values(["app_name"],ascending=False).groupby(["category"]).head(15)
print(dev_df)
duplicate_Dev = dev_df[dev_df.duplicated(['offered_by'],keep=False)]
fig = px.bar(duplicate_Dev, x="offered_by", y="app_name", color="category",labels={"offered_by":"offered_by","app_name":"app"}, barmode='stack')
fig.show()


df['category']=df['category'].fillna("Dummy")
#converting categorical columns to numerical using label encoder
le = LabelEncoder()
df['content_rating'].astype(str)
df['content_rating_num'] = le.fit_transform(df['content_rating'])

df['offered_by'].astype(str)
df['offered_by_num'] = le.fit_transform(df['offered_by'])

df['category'].astype(str)
df['category_by_num']=le.fit_transform(df['category'])


#Correlation matrix
corrMatrix = df[['reviews','installs','size','price in USD','content_rating_num','offered_by_num','year','month','category_by_num']].corr()
fig=plt.figure()
sns.heatmap(corrMatrix, annot=True,cmap="Blues")
plt.title("Heatmap of GooglePlay Apps Dataset")
plt.show()


X = df[['reviews','installs','size','price in USD','rate_5_pc','content_rating_num','offered_by_num']].values
Y = df['rating']

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.3)
#Linear Regression
model = LinearRegression()
model.fit(x_train,y_train)
prediction_lin = model.predict(x_test)
print("Linear Model:")
print("Mean squared error is :",np.round(mean_squared_error(y_test, prediction_lin),3))
print("r2 score :",np.round(r2_score(y_test,prediction_lin),2))
print("RMSE :",np.sqrt(mean_squared_error(y_test,prediction_lin)))

#Decision Tree Regressor
dtr = DecisionTreeRegressor(max_depth= 6)
dtr = dtr.fit(x_train, y_train)
prediction_dec = dtr.predict(x_test)
print("Decision Tree model:")
print("Mean squared error is :",np.round(mean_squared_error(y_test, prediction_dec),3))
print("r2 score :",np.round(r2_score(y_test,prediction_dec),2))
print("RMSE :",np.sqrt(mean_squared_error(y_test,prediction_dec)))

#xgboost regressor
xgr = XGBRegressor(max_depth =3)
xgr.fit(x_train, y_train)
prediction_xgr = xgr.predict(x_test)
print("xgboost regressor:")
print("Mean squared error is :",np.round(mean_squared_error(y_test, prediction_xgr),3))
print("r2 score :",np.round(r2_score(y_test,prediction_xgr),2))
print("RMSE :",np.sqrt(mean_squared_error(y_test,prediction_xgr)))




#https://pysimplegui.readthedocs.io/en/latest/#multiple-threads------Tcl_AsyncDelete: async handler deleted by the wrong thread
#https://stackoverflow.com/questions/48393336/pandas-groupby-sort-within-groups-retaining-multiple-aggregates
#https://datatofish.com/replace-values-pandas-dataframe/-----------for replacing values in a datframe
#https://www.geeksforgeeks.org/python-pandas-to_numeric-method/
#https://stackoverflow.com/questions/42961168/python-pandas-accessing-datetime-series-for-currency-conversion
#https://www.geeksforgeeks.org/convert-the-column-type-from-string-to-datetime-format-in-pandas-dataframe/
#https://stackabuse.com/text-translation-with-google-translate-api-in-python/
#https://pstblog.com/2016/10/04/stacked-charts
#https://medium.com/data-science-bootcamp/cool-plots-with-seaborn-barplot-with-hue-and-proportions-e59a66fafd15