package com.example.myapp1

import java.io.Serializable
import java.util.UUID

data class Trade(
    val id: String = UUID.randomUUID().toString(),
    var symbol: String = "",
    var type: String = "buy",
    var price: Double = 0.0,
    var quantity: Int = 0,
    var date: Long = System.currentTimeMillis(),
    var notes: String = "",
    val createdAt: Long = System.currentTimeMillis()
) : Serializable