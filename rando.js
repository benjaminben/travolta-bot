const fs = require("fs")
const f1 = "data/hairspray.txt"
const f2 = "data/canki-sutta.txt"
const of = "data/hairspray-sutta.txt"

let f1a = []
let f2a = []
let fin = []

fs.readFile(f1, "utf8", (err, data) => {
  f1a = data.split("\n")
  fin = f1a

  fs.readFile(f2, "utf8", (err, data) => {
    f2a = data.split("\n")

    f2a.forEach(b => {
      fin.splice(Math.floor(Math.random() * fin.length), 0, b)
    })

    fs.writeFile(of, fin.join("\n"), (err) => {
      console.log("done")
    })
  })
})
