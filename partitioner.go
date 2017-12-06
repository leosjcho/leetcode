package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"strings"
)

const golangQuestionHeaderPattern = `\/\*\s\d.*\s\*\/`
const pythonQuestionHeaderPattern = `['"]{3}\s\d.*\s['"]{3}`

/*
TODO - add URLs to questions
53_maximum_subarray.go
would exist at
https://leetcode.com/problems/maximum-subarray
const baseLeetcodeURL = "https://leetcode.com/problems/"
*/

type monolith struct {
	filename     string
	regexPattern string
	destDir      string
	fileExt      string
}

func getCode(filename string) ([]byte, error) {
	f, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0755)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}
	return buf, nil
}

func findMatchingRanges(codeStr string, questionPattern string) [][]int {
	r, _ := regexp.Compile(questionPattern)
	return r.FindAllStringIndex(codeStr, -1)
}

func formattedTitleForFile(t string, m *monolith) string {
	terms := strings.Fields(t)
	terms = terms[1 : len(terms)-1]
	title := strings.Join(terms, "_")
	commalessTitle := strings.Replace(title, ".", "", -1)
	return strings.ToLower(commalessTitle) + m.fileExt
}

func saveFile(title, data, path string) error {
	f, err := os.Create(path + title)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(data)
	if err != nil {
		return err
	}
	f.Sync()
	return nil
}

func saveFileWithRange(codeStr string, matchRange []int, nextStart int, m *monolith) error {
	title := formattedTitleForFile(codeStr[matchRange[0]:matchRange[1]], m)
	data := codeStr[matchRange[0]:nextStart]
	err := saveFile(title, data, m.destDir)
	if err != nil {
		return err
	}
	return nil
}

func nextStart(curIndex int, matchingRanges [][]int, codeStr string) int {
	if curIndex == len(matchingRanges)-1 {
		return len(codeStr)
	}
	return matchingRanges[curIndex+1][0]
}

func monoliths() []monolith {
	p := monolith{
		filename:     "python/leetcode.py",
		regexPattern: pythonQuestionHeaderPattern,
		destDir:      "python/",
		fileExt:      ".py",
	}
	g := monolith{
		filename:     "golang/leetcode.go",
		regexPattern: golangQuestionHeaderPattern,
		destDir:      "golang/",
		fileExt:      ".go",
	}
	return []monolith{g, p}
}

func main() {
	for _, m := range monoliths() {
		code, err := getCode(m.filename)
		if err != nil {
			log.Fatal(err)
		}
		codeStr := string(code)
		matchingRanges := findMatchingRanges(codeStr, m.regexPattern)
		for i, matchRange := range matchingRanges {
			ns := nextStart(i, matchingRanges, codeStr)
			err := saveFileWithRange(codeStr, matchRange, ns, &m)
			if err != nil {
				log.Fatalln(err)
			}
		}
		fmt.Printf("Wrote %v %v files into %v.\n", len(matchingRanges), m.fileExt, m.destDir)
	}

}
