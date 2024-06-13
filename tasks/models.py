from django.db import models
from django.contrib.auth.models import User

class Usuarios(models.Model):
    # Campos de texto
    nombre_usuario = models.CharField(max_length=30, unique=True)
    nombre = models.CharField(max_length=50)
    apellidos = models.CharField(max_length=50)
    email = models.EmailField(unique=True)
    telefono = models.CharField(max_length=20)
    fecha_nacimiento = models.DateField()
    ultimo_login = models.DateTimeField(auto_now=True)
    esta_activa = models.BooleanField(default=True)
    es_personal = models.BooleanField(default=False)
    OPCIONES_ROL = [
        ('Personal', 'Personal Use'),
        ('Enterprise', 'Enterprise'),
    ]
    roles = models.CharField(max_length=11, choices=OPCIONES_ROL, default='Personal')
    
class Task(models.Model):
    # campos
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    created = models.DateTimeField(auto_now_add=True)
    datecompleted = models.DateTimeField(null=True, blank=True)
    important = models.BooleanField(default=False)
    # cascate - si queres que si se borra el usuario se borre sus tareas se hace en cascada
    user = models.ForeignKey(User, on_delete=models.CASCADE)


    def __str__(self):
        return self.title + ' - by - ' + self.user.username
