from django import forms
from .models import Task, Usuarios

#definimos nuestros formularios personalizados
# con los campos del model de la db
class TaskForm(forms.ModelForm):
    class Meta:
        model = Task
        fields = ['title', 'description','important']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Write a Title'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'placeholder': 'Write a Description for the Task'}),
            'important': forms.CheckboxInput(attrs={'class': 'form-check-input text-center'})
        } 
        
class UsuarioForm(forms.ModelForm):
    class Meta:
        model = Usuarios
        # definimos los campos de la DB
        fields = [
            'nombre_usuario',
            'nombre',
            'apellidos',
            'email',
            'telefono',
            'fecha_nacimiento',
            'roles',
        ]
