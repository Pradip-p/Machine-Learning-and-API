from django.db import models

# Create your models here.
class Titanic(models.Model):
    PassengerId=models.CharField(max_length=50)
    Pclass=models.IntegerField(default=0)
    Name=models.CharField(max_length=50)
    Sex=models.CharField(max_length=50)
    Age=models.IntegerField()
    SibSp=models.IntegerField()
    Parch=models.IntegerField()
    Ticket=models.CharField(max_length=50)
    Fare=models.FloatField()
    Cabin=models.CharField(max_length=50, null=True, blank=True)
    Embarked=models.CharField(max_length=50)
    Survived=models.IntegerField(null=True, blank=True)
    def to_dict(self):
        return{
            'PassengerId':self.PassengerId,
            'Pclass':self.Pclass,
            'Name':self.Name,
            'Sex':self.Sex,
            'Age':self.Age,
            'SibSp':self.SibSp,
            'Parch':self.Parch,
            'Ticket':self.Ticket,
            'Fare':self.Fare,
            'Cabin':self.Cabin,
            'Embarked':self.Embarked,
            'Survived':self.Survived

        }