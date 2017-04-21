/*
448. Find All Numbers Disappeared in an Array
*/

class Solution {
    func findDisappearedNumbers(_ nums: [Int]) -> [Int] {
        /*
        var numMap:[Int:Bool] = [:]
        var missingNums:[Int] = []
        for i in 0..<nums.count {
            numMap[nums[i]] = true
        }
        if nums.count > 0 {
            for i in 1...nums.count {
                if numMap[i] == nil {
                    missingNums.append(i)
                }
            }
        }
        return missingNums
        */
        var numsmutable = nums
        var missingNums:[Int] = []
        for i in 0..<numsmutable.count {
            var val = abs(numsmutable[i]) - 1
            if numsmutable[val] > 0 {
                numsmutable[val] = -numsmutable[val]
            }
        }
        if numsmutable.count > 0 {
            for j in 1...numsmutable.count {
                if numsmutable[j-1] > 0 {
                    missingNums.append(j)
                }
            }
        }
        return missingNums
    }
}


/*
463. Island Perimeter
*/

class Solution {
    func islandPerimeter(_ grid: [[Int]]) -> Int {

    }
}

/*
457. Circular Array Loop
*/

class Solution {
    func circularArrayLoop(_ nums: [Int]) -> Bool {

    }
}
