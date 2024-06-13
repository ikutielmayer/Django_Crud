from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.contrib.auth import login, logout, authenticate
from django.db import IntegrityError
from .forms import TaskForm
from .forms import UsuarioForm
from .models import Task
from django.utils import timezone
from django.contrib.auth.decorators import login_required

def home(request):
    return render(request, 'home.html')

@login_required
def task_detail(request, task_id):
    if request.method == 'GET':
        task = get_object_or_404(Task, pk=task_id, user=request.user)
        form = TaskForm(instance=task)
        return render(request, 'task_detail.html',
                      {'task': task, 'form': form})
    else:
        try:
            task = get_object_or_404(Task, pk=task_id, user=request.user)
            form = TaskForm(request.POST, instance=task)
            form.save()
            return redirect('tasks')
        except ValueError:
            return render(request, 'task_detail.html',
                          {'task': task, 'form': form,
                           'errror': 'Error during update'})

@login_required
def complete_task(request, task_id):
    task = get_object_or_404(Task, pk=task_id, user=request.user)
    if request.method == 'POST':
        task.datecompleted = timezone.now()
        task.save()
        return redirect('tasks')

@login_required
def delete_task(request, task_id):
    task = get_object_or_404(Task, pk=task_id, user=request.user)
    if request.method == 'POST':
        task.delete()
        return redirect('tasks')

@login_required
def profile(request):
    if request.method == 'POST':
         try:
            form = UsuarioForm(request.POST)
            if form.is_valid():
                form.save()
                login(request, user)
                return redirect('tasks')
         except IntegrityError:
                return render(request, 'register.html', {
                    'form': UserCreationForm,
                    'error': 'Please Check you information'})
                # return HttpResponse('UserName already exist')
    else:
        form = UsuarioForm()
    return render(request, 'register.html', {'form': form})
    
def register(request):
    if request.method == 'POST':
         try:
            form = UsuarioForm(request.POST)
            if form.is_valid():
                form.save()
                login(request, user)
                return redirect('tasks')
         except IntegrityError:
                return render(request, 'register.html', {
                    'form': UserCreationForm,
                    'error': 'Please Check you information'})
                # return HttpResponse('UserName already exist')
    else:
        form = UsuarioForm()
    return render(request, 'register.html', {'form': form})

def signup(request):

    if request.method == 'GET':
        print('Enviando Datos...')
        return render(request, 'signup.html', {'form': UserCreationForm})
    else:
        print(request.POST)
        print('Obteniendo Datos...')
        if request.POST['password1'] == request.POST['password2']:
            try:
                user = User.objects.create_user(
                    username=request.POST['username'], password=request.POST['password1'])
                user.save()
                login(request, user)
                return redirect('tasks')
            except IntegrityError:
                return render(request, 'signup.html', {
                    'form': UserCreationForm,
                    'error': 'UserName already exist'})
                # return HttpResponse('UserName already exist')

        return render(request, 'signup.html', {
            'form': UserCreationForm,
            'error': 'Password do Not match'})

@login_required
def tasks(request):
    # tasks = Task.objects.all()  # Devuelve todas las tareas de la DB
    tasks = Task.objects.filter(user=request.user, datecompleted__isnull=True)
    return render(request, 'tasks.html', {'tasks': tasks})

@login_required
def tasks_completed(request):
    tasks = Task.objects.filter(user=request.user, datecompleted__isnull=False).order_by('-datecompleted')
    return render(request, 'tasks.html', {'tasks': tasks})


@login_required
def create_task(request):
    if request.method == 'GET':
        return render(request, 'create_task.html', {
            'form': TaskForm,
            'error': 'Revise los Datos'})
    else:
        # print(request.POST)
        try:
            form = TaskForm(request.POST)
            new_task = form.save(commit=False)
            new_task.user = request.user
            print(new_task)
            new_task.save()
           # return render(request, 'tasks.html')
            return redirect('tasks')
        except ValueError:
            return render(request, 'create_task.html', {
                'form': TaskForm,
                'error': 'Error: Revise los Datos'})

@login_required
def signout(request):
    logout(request)
    return redirect('home')


def signin(request):
    if request.method == 'GET':
        print('Entro con Login...')
        return render(request, 'signin.html', {'form': AuthenticationForm})
    else:
        # print(request.POST)
        user = authenticate(
            request, username=request.POST['username'], password=request.POST['password'])
        if user is None:
            return render(request, 'signin.html', {'form': AuthenticationForm, 'error': 'UserName or Password is Incorrect!'})
        else:
            login(request, user)
            return redirect('tasks')
