import requests


# -- Get dataset --
data = requests.get("https://www.scss.tcd.ie/Doug.Leith/CSU44061/week3.php").text
open("data","x")
open("data","w").write(data)
