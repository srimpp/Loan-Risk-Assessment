"""final URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from main import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.MainFunc),
    path('about', views.AboutUs),
    path('data', views.DataRecap),
    path('model', views.ListModels),
    path('predict', views.ModelPred),
    path('test', views.Test),
    
    path('model_DecisionTree', views.model_DecisionTree),
    path('model_KNN', views.model_KNN),
    path('model_MLP', views.model_MLP),
    path('model_SVM', views.model_SVM),
    path('model_Tensor', views.model_Tensor),
    
    path('model_LGBMClassifier', views.model_LGBMClassifier),
    path('model_LogisticRegression', views.model_LogisticRegression),
    path('model_NaiveBayes', views.model_NaiveBayes),
    path('model_RandomForest', views.model_RandomForest),
    path('model_XGBClassifier', views.model_XGBClassifier),
    
]
