from django.forms import ModelForm
from .models import Task

#definimos nuestros formularios personalizados
# con los campos del model de la db
class TaskForm(ModelForm):
    class Meta:
        model = Task
        fields = ['title', 'description','important']