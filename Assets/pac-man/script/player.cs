using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class player : PlayObject { 
    public float moveSpeed=0;
    public Vector4 input;
    public GameObject plane;
    public int flagE1 = 0;
    public int flagE2 = 0;
    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        input.Set(0, 0, 0, 0);
        if (Input.GetKey(KeyCode.W))
        {
            transform.Translate(Vector3.forward * moveSpeed * Time.deltaTime);
            input.x = 1;
        }
        if (Input.GetKey(KeyCode.S))
        {
            transform.Translate(Vector3.back * moveSpeed * Time.deltaTime);
            input.y = 1;
        }
        if (Input.GetKey(KeyCode.A))
        {
            transform.Translate(Vector3.left * moveSpeed * Time.deltaTime);
            input.z = 1;
        }
        if (Input.GetKey(KeyCode.D))
        {
            transform.Translate(Vector3.right * moveSpeed * Time.deltaTime);
            input.w = 1;
        }

        // KeepInsideViewport();

    }
    
    // void KeepInsideViewport()
    // {
    //     Vector3 viewPos = Camera.main.WorldToViewportPoint(transform.position);
    //     viewPos.x = Mathf.Clamp(viewPos.x, 0.05f, 0.95f);
    //     viewPos.y = Mathf.Clamp(viewPos.y, 0.05f, 0.95f);
    //     transform.position = Camera.main.ViewportToWorldPoint(viewPos);
    // }


    private void OnTriggerStay(Collider collision)
    {
        if (collision.name == "enemy1")
        {
            if (flagE2 == 0)
            {
                plane.GetComponent<Renderer>().material.SetColor("_CellColor", Color.red);
            }
            flagE1 = 1;
        }

        if (collision.name == "enemy2")
        {
            if (flagE1 == 0)
            {
                plane.GetComponent<Renderer>().material.SetColor("_CellColor", Color.red);
            }
            flagE2 = 1;
        }
    }

    private void OnTriggerExit(Collider collision)
    {
        if (collision.name == "enemy1")
        {
            flagE1 = 0;
            if (flagE2 == 0)
            {
                plane.GetComponent<Renderer>().material.SetColor("_CellColor", Color.grey);
            }
        }

        if (collision.name == "enemy2")
        {
            flagE2 = 0;
            if (flagE1 == 0)
            {
                plane.GetComponent<Renderer>().material.SetColor("_CellColor", Color.grey);
            }
        }
    }
    
}
