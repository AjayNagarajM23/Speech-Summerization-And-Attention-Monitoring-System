from django.db import models

# Create your models here.

import os
from uuid import uuid4

def path_and_rename(instance, filename):
    upload_to = 'videos'
    ext = filename.split('.')[-1]
    filename = "test_vid."+ext
    # return the whole path to the file
    return os.path.join(upload_to, filename)

class vid(models.Model):
    video = models.FileField(upload_to=path_and_rename)  


