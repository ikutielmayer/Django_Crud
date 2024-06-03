from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.contrib.auth import login, logout, authenticate
from django.db import IntegrityError
from .forms import TaskForm

# Create your views here.


def home(request):
    return render(request, 'home.html')


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


def tasks(request):
    return render(request, 'tasks.html')


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
            return render(request, 'tasks.html')
       except ValueError:
          return render(request, 'create_task.html', {
                    'form': TaskForm,
                    'error': 'Error: Revise los Datos'})
          
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

   