/*
292. Nim Game
*/

/*
if n == 1, 2, or 3, then we win!
if n == 4, what happens? we lose!
if n == 5? we win!
if n == 6? we win!
if n == 7? we win!
if n == 8? we lose!
etc...
*/

func canWinNim(n int) bool {
	return n%4 != 0
}

