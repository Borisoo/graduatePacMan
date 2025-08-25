using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class goal_light : MonoBehaviour
{
    private Vector3 new_pos;
    public float lenth=10;
    public float width=10;
    public PlayObject player;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void OnTriggerEnter(Collider other)
    {

       if (other.name== "Point Light")
        {
            new_pos.x = Random.Range(lenth, -lenth) + player.transform.position.x;
            new_pos.y = player.transform.position.y;
            new_pos.z = Random.Range(width, -width) + player.transform.position.z;
            other.transform.position = new_pos;
        }
    }
}
