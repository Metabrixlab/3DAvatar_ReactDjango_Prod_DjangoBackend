from django.db import models
from django.forms import ModelForm
from django import forms

# Create your models here.  
    
class image(models.Model):
    title = models.CharField(max_length=30, default="default_title")
    img = models.ImageField(upload_to="original/", default="original.png")
    modified = models.ImageField(upload_to="modified/", default="modified.png")
    
    def __str__(self) -> str:
        return self.title
    
class image_form(ModelForm):    
    class Meta:
        model = image
        fields = ["img"]
        labels = {"img":"",}
        widgets = {"img":forms.FileInput(attrs={"id" : "imageInput"})}