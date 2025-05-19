from django.db import models

# Create your models here.
class Densenet_model(models.Model):
    S_No = models.AutoField(primary_key=True)
    model_accuracy = models.CharField(max_length=10, default='99.8') 

    class Meta:
        db_table = "Densenet_model"

class MobileNet_model(models.Model):
    S_No = models.AutoField(primary_key=True)
    model_accuracy = models.CharField(max_length=10, default='99.7')  

    class Meta:
        db_table = "MobileNet_model"

class cnn_model(models.Model):
    S_No = models.AutoField(primary_key=True)
    model_accuracy = models.CharField(max_length=10, default='99.8')  

    class Meta:
        db_table = "cnn_model"



class Dataset(models.Model):
    title = models.CharField(max_length=200)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True) 
  
    class Meta:
        db_table = 'dataset_table'