from django.shortcuts import render
from rest_framework import viewsets
from . models import Titanic
from api.serializers import TitanicSerializer
from . import TiML
from rest_framework.response import Response
from django.contrib.auth.models import User


# Create your views here.
class TitanicViewSet(viewsets.ModelViewSet):
    queryset=Titanic.objects.all().order_by('-id')
    serializer_class=TitanicSerializer
    def create(self, request, *args,**kwargs):
          super(viewsets.ModelViewSet, self).create(request,*args,**kwargs)
          # viewsets.ModelViewSet.create(request,*args,**kwargs)
          ob=Titanic.objects.latest('id')
          sur=TiML.pred(ob)
          return Response({"status":"Success","Survived":sur,'tmp':args})
          

         

