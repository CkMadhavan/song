#!/bin/sh
read -p 'Enter Commit Name : ' name
git add .
git commit -m $name
git push heroku master
git push -u origin master 
