from rest_framework import serializers
from api.models import Titanic
from django.contrib.auth.models import User




class TitanicSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model= Titanic
        fields='__all__'

