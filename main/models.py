from django.db import models


# Create your models here.
class cache(models.Model):
    path = models.TextField(primary_key=True, unique=True, blank=False)
    vector = models.BinaryField()
    color = models.BinaryField()
    shape = models.BinaryField()
    texture = models.BinaryField()
    cnn = models.BinaryField()
